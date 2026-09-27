**006**

**009**

**019**

**024**

**029 030**

**032**

**034**

**036**

# HOTTEL ZONE PHYSICS-CONSTRAINED NETWORKS FOR FURNACES

Anonymous authors Paper under double-blind review

### ABSTRACT

This paper investigates a novel approach to improve the temperature profile prediction of furnaces in foundation industries, crucial for sustainable manufacturing. While existing methods like the Hottel Zone model are accurate, they lack real-time inference capabilities. Deep learning methods excel in speed and prediction but require careful generalization for real-world applications. We propose a regularization technique that leverages the Hottel Zone method to make deep neural networks physics-aware, improving prediction accuracy for furnace temperature profiles. Our approach demonstrates effectiveness on various neural network architectures, including Multi-Layer Perceptrons (MLP), Long Short-Term Memory (LSTM), Extended LSTM (xLSTM) and Kolmogorov-Arnold Networks (KANs). We also discussion the data generation involved.

# 1 INTRODUCTION

Majority of economically relevant industries (automobiles, machinery, construction, household appliances, chemicals, etc) are dependent on the Foundation Industries (FIs) that provide crucial and foundational materials like glass, metals, cement, ceramics, bulk chemicals, paper, steel, etc. FIs are heavy revenue and employment drivers, for instance, FIs in the United Kingdom (UK) economy are worth £52B [\(EPSRC report\)](#page-10-0), employ 0.25 million people, and comprise over 7000 businesses [\(IOM3](#page-11-0) [report\)](#page-11-0). However, despite their economic significance, the FIs leverage energy-intensive methods within their furnaces. This makes FIs major industrial polluters and the largest consumers of natural resources across the globe. For example, in the UK, they produce 28 million tonnes of materials per year, and generate 10% of the entire UK's CO<sup>2</sup> emissions [\(EPSRC report;](#page-10-0) [IOM3 report\)](#page-11-0). Similarly, in China, the steel industry accounted for 15% of the total energy consumption, and 15.4% of the total CO<sup>2</sup> emissions [\(Zhang et al.,](#page-13-0) [2018;](#page-13-0) [Liang et al.,](#page-11-1) [2020\)](#page-11-1). These numbers put a challenge for the FIs in meeting our commitment to reduce net Green-House Gas (GHG) emissions, globally.

With a closer look at any process industry (e.g., steel industry), one can observe that at the core, lies the process of conversion of materials (e.g., iron) into final products. This is done using a series of unit processes [\(Yu et al.,](#page-13-1) [2007\)](#page-13-1) involving steps such as dressing, sintering, smelting, casting, rolling, etc (see [Qin et al.](#page-12-0) [\(2022\)](#page-12-0) for an illustration). The equipment in such process industries operates in high-intensity environments (e.g., high temperature), and has bottleneck components such as reheating furnaces, which require complex restart processes post-failure. This causes additional labor costs and energy consumption. Thus, for sustainable manufacturing, it is important to monitor the temperature profile, and thus, the operating status of the furnaces. [\(Hu et al.,](#page-11-2) [2019\)](#page-11-2) have shown promise in achieving notable fuel consumption reduction by reducing the overall heating time.

[Yuen & Takara](#page-13-2) [\(1997\)](#page-13-2) in their study, have proved the elegance and superiority of the Hottel Zone method over counterparts to model the physical phenomenon of Radiative Heat Transfer (RHT) in high-temperature processes. [Hu et al.](#page-10-1) [\(2016\)](#page-10-1) proposed a computational model workflow based on the Hottel Zone method, and showed superiority over surrogate computational alternatives in terms of predictive performance. However, none of these approaches are suitable for real-time inference in modeling a furnace temperature profile. Deep Learning (DL) based neural network methods excel in achieving superior predictive performance and speed. Nonetheless, their generalization capabilities require special attention, particularly in critical real-world applications.

In our work, we propose to revisit the Hottel Zone method and devise a novel regularization technique that could be used as a plug-and-play module to make a neural network physics-constrained (or

**059**

**061**

**064**

**067**

**069 070 071**

**074**

**079**

**089 090 091**

**094**

**104 105 106**

physics-aware) with regard to the underlying phenomena of high-temperature processes in furnaces. We show that for a time-step in a furnace, given a certain set of input entities, we could predict the desired output temperature entities more accurately (in terms of regression metrics) using our regularization technique, as opposed to using a vanilla neural network. We demonstrate the prowess of our proposal on different types of neural network architectures: Multi-Layer Perceptron (MLP) or feed-forward networks, sequential models such as Long Short-Term Memory (LSTM) based Recurrent Neural Networks (RNNs), as well as recently proposed Kolmogorov-Arnold Networks (KANs) and Extended LSTM (xLSTM).

This work makes two key contributions: Tensor-based Reformulation and Physics-Aware Neural Networks: We reformulate the Hottel Zone Method's Directed Flux Areas (DFAs) and Energy Balance (EB) equations in tensor format, enabling neural network training. We further introduce a novel regularization technique that imbues the network with physics-awareness. Extensive Experimental Validation: We comprehensively validate the proposed approach using various neural network architectures. To this end, we suggest a dataset and benchmarking protocol (details provided in Section [A.8\)](#page-23-0). A github repository is maintained at <https://github.com/> to facilitate real-time updates to the same as and when made.

Numerous real-world applications, including chemical reactors [\(Feng & Han,](#page-10-2) [2012\)](#page-10-2), solar energy [\(Muhich et al.,](#page-12-1) [2016;](#page-12-1) [Marti et al.,](#page-11-3) [2015\)](#page-11-3), and 3D printing [\(Tran & Lo,](#page-12-2) [2018;](#page-12-2) [Zhou et al.,](#page-13-3) [2009\)](#page-13-3), involve high-temperature processes exceeding 700◦C. These processes rely heavily on Radiative Heat Transfer (RHT) as a dominant mechanism alongside conduction and convection. Notably, RHT remains crucial for thermal transport even in vacuum conditions encountered in astronomical applications. We envision that our learnings could perhaps be extended to those applications with bespoke approaches.

Due to space constraints, we have limited the length of the introduction section. Please refer to Section [A.1](#page-13-4) for a more detailed discussion, particularly regarding the motivation behind our research.

# 2 RELATED WORK

In Section [A.2,](#page-16-0) we provide a detailed discussion of related works. Due to space limitations, we will focus here on how our approach significantly differs from existing methods.

- 1. View factor methods: Existing methods [Ebrahimi et al.](#page-10-3) [\(2013\)](#page-10-3); [Melot et al.](#page-12-3) [\(2011\)](#page-12-3); [Hu](#page-11-4) [et al.](#page-11-4) [\(2018\)](#page-11-4); [Li](#page-11-5) [\(2005\)](#page-11-5) simplify the modeling area and are geometry-specific. We propose a generic, geometry-agnostic model encompassing all exchange areas (radiation transfer interfaces).
- 2. Neural network methods: Existing methods [Yuen](#page-13-5) [\(2009\)](#page-13-5); [Tausendschon & Radl](#page-12-4) ¨ [\(2021\)](#page-12-4); [Garc´ıa-Esteban et al.](#page-10-4) [\(2021\)](#page-10-4); [Zhai & Zhou](#page-13-6) [\(2020\)](#page-13-6); [Zhai et al.](#page-13-7) [\(2023\)](#page-13-7); [Halme Stahlberg](#page-10-5) ˚ [\(2021\)](#page-10-5); [de Souza Lima et al.](#page-10-6) [\(2023\)](#page-10-6); [Liao et al.](#page-11-6) [\(2009\)](#page-11-6); [Hwang et al.](#page-11-7) [\(2019\)](#page-11-7); [Chen et al.](#page-10-7) [\(2022\)](#page-10-7); [Bao et al.](#page-9-0) [\(2023\)](#page-9-0) often use simple MLPs, which lack generalization due to limited physics understanding. We introduce a Physics-constrained Neural Network (PCNN) framework that outperforms MLP and can be applied to other architectures like LSTM, KAN, xLSTM.
- 3. Furnace temperature profiling: Existing methods [Kim & Huh](#page-11-8) [\(2000\)](#page-11-8); [Kim](#page-11-9) [\(2007\)](#page-11-9); [Jang](#page-11-10) [et al.](#page-11-10) [\(2010\)](#page-11-10); [Tang et al.](#page-12-5) [\(2017\)](#page-12-5); [Nguyen et al.](#page-12-6) [\(2014\)](#page-12-6); [Hu et al.](#page-10-8) [\(2017\)](#page-10-8); [Ban et al.](#page-9-1) [\(2023\)](#page-9-1); [Li et al.](#page-11-11) [\(2023\)](#page-11-11); [Zanoli et al.](#page-13-8) [\(2023\)](#page-13-8); [Yu et al.](#page-13-9) [\(2022\)](#page-13-9) focus on specific regions, while our method targets complete furnace temperature profiling, including gas zones, furnace walls, and slab surfaces. Our utilized data is more holistic. Existing neural methods in this category also lack physics awareness.
- 4. PINNs: Compared to the existing body of Physics-Informed Neural Network (PINN) literature [Raissi et al.](#page-12-7) [\(2019\)](#page-12-7); [Karniadakis et al.](#page-11-12) [\(2021\)](#page-11-12); [Drgona et al.](#page-10-9) ˇ [\(2021\)](#page-10-9); [Shen et al.](#page-12-8) [\(2023\)](#page-12-8); [Cai et al.](#page-9-2) [\(2021\)](#page-9-2); [Kim et al.](#page-11-13) [\(2022\)](#page-11-13); [Zhao et al.](#page-13-10) [\(2020\)](#page-13-10); [He et al.](#page-10-10) [\(2021\)](#page-10-10); [Boca de](#page-9-3) [Giuli](#page-9-3) [\(2023\)](#page-9-3); [Han et al.](#page-10-11) [\(2023\)](#page-10-11); [Bunning et al.](#page-9-4) ¨ [\(2022\)](#page-9-4); [Park](#page-12-9) [\(2022\)](#page-12-9); [Wang et al.](#page-12-10) [\(2023\)](#page-12-10); [Lahariya et al.](#page-11-14) [\(2022\)](#page-11-14); [Jing et al.](#page-11-15) [\(2023\)](#page-11-15), we propose a novel variant specifically designed for zone method based modeling in reheating furnaces. Our approach is the first to utilize physics-constrained regularizers based on the zone method for temperature prediction. It requires minimal data (input-output pairs) and makes no geometry assumptions. Our data creation method is holistic and unique, encompassing all exchange areas. Our method, as we

**114 115**

**117**

**119**

**127**

**129 130**

**134**

**136**

will see later, is based on a set of simultaneous equations to incorporate physics-awareness, and directly does not involve a differential equation. Thus, we call it a physics-constrained method, though PINN could be also used philosophically.

#### 3 PROPOSED METHOD

#### 3.1 BACKGROUND

The Hottel Zone method subdivides a furnace into zones (volumes and surfaces) to predict Radiative Heat Transfer (RHT). Volume and Gas (G) zone is used interchangeably. Surface (S) zones are of two types, SF: furnace and SO: obstacle (e.g., slabs that are heated). Each zone has a uniform temperature. Sets of Energy-Balance (EB) equations govern radiation exchange between zones, considering incoming and outgoing radiation fluxes. These equations are iteratively updated to obtain the entire furnace's temperature profile. Following are the key concepts:

- 1. Total Exchange Areas (TEAs): Pre-computed values representing the total area for radiation exchange between zone pairs (SS: surface-surface, SG/GS: surface-gas, GG: gas-gas).
- 2. Directed Flux Areas (DFAs): Derived from TEAs and used to calculate radiant exchange between zone pairs at each step of the zone method.
- 3. Weighted Sum of Grey Gases (WSGG) model: Handles non-grey gases by representing them as a mixture of grey gases and a clear gas.

#### 3.2 EXCHANGE AREA CALCULATION

The first step in the Zone method involves computation of Exchange Factors [\(Yuen & Takara,](#page-13-2) [1997\)](#page-13-2). The exchange factor among a pair of volume zones V<sup>i</sup> and V<sup>j</sup> is expressed as:

$$g_i g_j = \int_{V_i} \int_{V_j} \frac{k_i k_j e^{-\tau} dV_i dV_j}{\pi r^2} \quad (1)$$

Physically, it represents the energy radiated from V<sup>i</sup> and absorbed/ scattered by V<sup>j</sup> . Here, k denotes the respective extinction coefficient, τ is the optical thickness among differential volume elements dV<sup>i</sup> and dV<sup>j</sup> , and r = p (x<sup>i</sup> − x<sup>j</sup> ) <sup>2</sup> + (y<sup>i</sup> − y<sup>j</sup> ) <sup>2</sup> + (z<sup>i</sup> − z<sup>j</sup> ) <sup>2</sup>. Now, let n<sup>i</sup> and n<sup>j</sup> respectively be unit normal vectors of dA<sup>i</sup> and dA<sup>j</sup> (corresponding to two surface zones A<sup>i</sup> and A<sup>j</sup> ). Then, the exchange factors gis<sup>j</sup> (between volume zone V<sup>i</sup> and surface zone A<sup>j</sup> ) and sis<sup>j</sup> (between surface zone A<sup>i</sup> and surface zone A<sup>j</sup> ), can be expressed as:

$$g_i s_j = \int_{V_i} \int_{A_j} \frac{k_i |\mathbf{n}_j \cdot \mathbf{r}| e^{-\tau} dV_i dA_j}{\pi r^3}; s_i s_j = \int_{A_i} \int_{A_j} \frac{|\mathbf{n}_i \cdot \mathbf{r}| |\mathbf{n}_j \cdot \mathbf{r}| e^{-\tau} dA_i dA_j}{\pi r^4} \quad (2)$$

Numerical evaluation of the above equations being complex, has led to analytical approximations, by considering an enclosure as a cube-square system, i.e, by representing a volume as a cube, and a surface as a square. This facilitates the tabulation of a "generic" set of exchange factors, which are applicable for most practical industrial geometries, using an updated Monte-Carlo based Ray-Tracing (MCRT) algorithm [\(Matthew et al.,](#page-12-11) [2014\)](#page-12-11). To this end, such pre-computed generic values are refered to as Total Exchange Areas (TEA), and we denote them by: GiS<sup>j</sup> , SiS<sup>j</sup> , GiG<sup>j</sup> and SiG<sup>j</sup> . Here, SiG<sup>j</sup> = GiS<sup>j</sup> . Note that throughout the text, G(or g) and S(or s) shall indicate terms corresponding to Gas/Volume, and Surface respectively.

#### 3.3 INTRODUCING TENSOR NOTATIONS FOR HOTTEL ZONE METHOD BASED NEURAL NETWORK

To account for our formulation of a neural network based approach, we first introduce the following four tensors to collectively represent the above TEAs: GS ∈ R |G|×|S|×|Ng| , SS ∈ R |S|×|S|×|Ng| , GG ∈ R |G|×|G|×|Ng| , SG ∈ R |S|×|G|×|Ng| . Here, |G|, |S| respectively denote the number of gas/ volume zones, and number of surface zones. In practice, |Ng| gases representing real gas medium are used, and hence, a third dimension has also been used in the above tensors. As discussed above, TEAs are pre-computed constants, used as inputs to our model. Slightly abusing notations, we can refer to a TEA by considering only the first two dimensions (for a pair of zones).

**166 167**

**169**

**171**

**204**

**206**

![](_page_3_Diagram_0.jpeg)

Figure 1: Derivation of matrix forms of the DFA terms (using GS as reference).

The next step is to compute the Radiation Exchange factors, or the Directed Flux Areas (DFA), considering radiating gas medium through a Weighted Sum of the mixed Grey Gases (WSGG) model [\(Hu et al.,](#page-10-1) [2016\)](#page-10-1):

$$G_i^{\overleftarrow{}}G_j = \sum_{n=1}^{N_g} a_{g,n}(T_{g,j})(\overline{G_iG_j})_{k=k_n}; S_i^{\overleftarrow{}}S_j = \sum_{n=1}^{N_g} a_{s,n}(T_{s,j})(\overline{S_iS_j})_{k=k_n} \quad (3)$$

$$G_i^{\overleftarrow{S}_j} = \sum_{n=1}^{N_g} a_{s,n}(T_{s,j})(\overline{G_i S_j})_{k=k_n}; S_i^{\overleftarrow{G}_j} = \sum_{n=1}^{N_g} a_{g,n}(T_{g,j})(\overline{S_i G_j})_{k=k_n} \quad (4)$$

Here, ↼ indicates the direction of flow. Tg,j and Ts,j denote the temperatures for the j th volume and surface zones respectively, and are the values we want our model to predict (at each time step). Note that the collective representation of the DFAs can be expressed as: ↼ GS ∈ R |G|×|S| , ↼ SS ∈ R |S|×|S| , ↼ GG ∈ R |G|×|G| , ↼ SG ∈ R |S|×|G| . In Eq [\(3\)](#page-3-0)-[\(4\)](#page-3-1), the TEA terms correspond to a particular grey gas being used, for example, (GiG<sup>j</sup> )k=k<sup>n</sup> represents the TEA GiG<sup>j</sup> with the n th gas.

WSGG is a method used to represent the absorptivity/ emissivity of real combustion products with a mixture of a couple of grey gases plus a clear gas, i.e, the number of grey gases is equal to N<sup>g</sup> − 1.

For each gas indexed by n, we have a set of pre-computed correlation coefficients {bi+1,n} N<sup>g</sup> <sup>i</sup>=0 for both gas and surface related coefficients, and an absorption coefficient kg,n. Then, the weighting coefficient ag,n(Tg,j ) (for gas-zone temperatures) and the weighting coefficient as,n(Ts,j ) (for surface-zone temperatures) can be expressed as a Nth <sup>g</sup> order polynomial in Tg,j (or Ts,j ):

$$a_{g,n}(T_{g,j}) = \sum_{i=0}^{N_g} b_{i+1,n} T_{g,j}^i; a_{s,n}(T_{s,j}) = \sum_{i=0}^{N_g} b_{i+1,n} T_{s,j}^i \quad (5)$$

Using [\(3\)](#page-3-0), [\(4,](#page-3-1) [\(5\)](#page-3-2), and with GS as a reference, we make use of Figure [1](#page-3-3) to illustrate the derivation of a compact matrix form for computing a DFA term efficiently for getting training samples of a neural network. Let, (GS)<sup>n</sup> be the n th slice of GS along the third dimension, and a<sup>n</sup> = ˜bn(tS). broadcast(a ⊤ n ) reshapes a ⊤ n to the same dimension as (GS)n, i.e., <sup>R</sup> |G|×|S| . t<sup>S</sup> ∈ <sup>R</sup> |S| is a vector containing all the surface zone temperatures (in a time step), such that its j th entry tS(j) = Ts,j . The j th entry an(j) of a<sup>n</sup> ∈ <sup>R</sup> |S| is computed using the function ˜b<sup>n</sup> with the correlation coefficients {bi+1,n} N<sup>g</sup> <sup>i</sup>=0 as the parameters, and by following eq [\(5\)](#page-3-2). We can also assume similar vector containing all gas zone temperatures (in a time step) t<sup>G</sup> ∈ <sup>R</sup> |G| , with j th entry tG(j) = Tg,j .

**224**

**236 237**

**254**

**256**

**259**

![](_page_4_Diagram_0.jpeg)

Figure 2: Derviation of the matrix forms of the EBV equations for physics based regularizers.

Then, the DFA terms related to gas-zone temperatures can be expressed as:

$$\tilde{\mathbf{G}}\mathbf{S} = \sum_{n=1}^{N_g} (\overline{GS})_n \odot \text{broadcast}(\mathbf{a}_n^\top); \tilde{\mathbf{G}}\mathbf{G} = \sum_{n=1}^{N_g} (\overline{GG})_n \odot \text{broadcast}(\tilde{b}_n(\mathbf{t}_G)^\top). \quad (6)$$

and, the DFA terms related to surface-zone temperatures can be expressed as:

$$\bar{S}S = \sum_{n=1}^{N_g} (\bar{S}S)_n \odot \text{broadcast}(\tilde{b}_n(t_S)^\top); \bar{S}G = \sum_{n=1}^{N_g} (\bar{S}G)_n \odot \text{broadcast}(\tilde{b}_n(t_G)^\top). \quad (7)$$

#### 3.4 ENERGY-BALANCE BASED PHYSICS-REGULARIZATION

With the above DFA terms at our disposal, we can compute the gas/volume and surface zone temperatures at each time step of furnace operation by respectively using Energy-Balance Volume (EBV) and Energy-Balance Surface (EBS) equations. EBV and EBS are a set of simulataneous equations to capture the governing physics of RHT [Hu et al.](#page-10-1) [\(2016\)](#page-10-1). Figure [2](#page-4-0) visually illustrates computation of the terms g(g)arr, s(g)arr and gleave involved in the EBV equation to compute the gas zone temperatures of a time step.

Let, g(g)arr ∈ <sup>R</sup> <sup>|</sup>G<sup>|</sup> be a vector whose i th entry represents the amount of radiation arriving at the i th gas zone from all the other gas zones, s(g)arr ∈ <sup>R</sup> |G| , a vector whose i th entry represents the amount of radiation arriving at the i th gas zone from all the other surface zones, gleave ∈ <sup>R</sup> |G| , a vector whose i th entry represents the amount of radiation leaving the i th gas zone, and h<sup>g</sup> ∈ <sup>R</sup> |G| a heat term. Also, let Tg,j (or Tg) and Ts,j (or Ts) denote the j th gas and surface zone temperatures respectively. Then, following EBV equations, the i th entries of g(g)arr, s(g)arr, gleave and h<sup>g</sup> can be computed as:

$$g_{(g)arr}(i) = \sum_j^{|G|} \mathbf{G}_i^\top \mathbf{G}_j \sigma T_{g,j}^4; \quad s_{(g)arr}(i) = \sum_j^{|S|} \mathbf{G}_i^\top \mathbf{S}_j \sigma T_{s,j}^4 \quad (8)$$

$$g_{leave}(i) = \sum_n^{|N_g|} a_{g,n}(T_{g,i}) k_{g,n} \sigma V_i T_{g,i}^4, \quad \mathbf{h}_g(i) = -(\dot{Q}_{conv})_i + (\dot{Q}_{fuel,net})_i + (\dot{Q}_a)_i + \mathbf{q}_i$$

Here, the constants (known apriori) (Q˙ conv)<sup>i</sup> , (Q˙ fuel,net)<sup>i</sup> , and (Q˙ <sup>a</sup>)<sup>i</sup> respectively denote the convection heat transfer, heat release due to input fuel, and thermal input from air/ oxygen. An enthalpy vector q ∈ R |G| is computed using the flow-pattern obtained via polynomial curve fitting during simulation. σ is the Stefan-Boltzmann constant, V<sup>i</sup> is volume of i th gas zone.

Let, s(s)arr ∈ <sup>R</sup> |S| , be a vector whose i th entry represents the amount of radiation arriving at the i th surface zone from all the other surface zones, g(s)arr ∈ <sup>R</sup> |S| , a vector whose i th entry represents

**289 290 291**

**294**

**301**

**304**

**306**

**309**

**314 315**

**318 319**

**321**

the amount of radiation arriving at the i th surface zone from all the other gas zones, sleave ∈ <sup>R</sup> |S| , a vector whose i th entry represents the amount of radiation leaving the i th surface zone, and h<sup>s</sup> ∈ <sup>R</sup> |S| a heat term. Then, following EBS equations, the i th entries of s(s)arr, g(s)arr, sleave and h<sup>s</sup> can be computed as:

$$\begin{aligned} \mathbf{s}_{(s)arr}(i) &= \sum_j^{|S|} \mathbf{S}_i^{\top} \mathbf{S}_j \sigma \mathbf{T}_{s,j}^4; & \mathbf{g}_{(s)arr}(i) &= \sum_j^{|G|} \mathbf{S}_i^{\top} \mathbf{G}_j \sigma \mathbf{T}_{g,j}^4 \\ \mathbf{s}_{leave}(i) &= A_i \epsilon_i \sigma \mathbf{T}_{s,i}^4; & \mathbf{h}_s(i) &= A_i (\dot{\mathbf{q}}_{conv})_i - \dot{\mathbf{Q}}_{s,i} \end{aligned} \quad (9)$$

For a surface zone i, the constants (known apriori) Ai( ˙qconv)<sup>i</sup> and Q˙ s,i respectively denote the heat flux to the surface by convection and heat transfer from it to the other surfaces. Here, A<sup>i</sup> is the area, and ϵ<sup>i</sup> is the emissivity of the i th surface zone.

The calculated terms in the Energy-Balance (EB) equations represent the heat entering and leaving each zone. In simpler terms, these equations ensure an energy balance by placing all incoming heat terms on the left-hand side (LHS) and outgoing terms on the right-hand side (RHS). Leveraging these terms in an optimization framework allows us to minimize the difference between LHS and RHS. To achieve this, we introduce the following terms:

$$\begin{aligned} \mathbf{v}_g &= (\mathbf{g}_{(g)arr} + \mathbf{s}_{(g)arr} - 4\mathbf{g}_{leave} + \mathbf{h}_g) \in \mathbb{R}^{|G|} \\ \mathbf{v}_s &= (\mathbf{s}_{(s)arr} + \mathbf{g}_{(s)arr} - \mathbf{s}_{leave} + \mathbf{h}_s) \in \mathbb{R}^{|S|} \end{aligned} \quad (10)$$

Here, |G|/|S| denotes the number of Gas/ Surface zones. Intuitively, v<sup>g</sup> and v<sup>s</sup> are vector representatives corresponding to EBV and EBS. Let, λebv, λebs > 0 are hyper-parameters corresponding to Lebv and Lebs, such that Lebv=||normalize(vg)||<sup>2</sup> is our proposed regularizer term corresponding to the EBV. Similarly, Lebs=||normalize(vs)||<sup>2</sup> 2 is our proposed regularizer term corresponding to the EBS. We use: normalize(v) = v/max(v), where max(v) is the maximum value from among all components in v.

The core idea is to leverage the Energy Balance (EB) equations, which represent well-established physical laws governing heat transfer in the furnace. These equations enforce a balance between incoming and outgoing heat for each zone. The vectors v<sup>g</sup> and v<sup>s</sup> capture the residuals between the incoming and outgoing heat terms in the EB equations for gas (g) and surface (s) zones, respectively. By minimizing the L2 norm of these residuals (after normalization), we are essentially penalizing the network for deviating significantly from the physical constraints imposed by the EB equations. This encourages the network to learn temperature profiles that adhere to these well-defined energy balances.

Minimizing the L2 norm encourages the network to drive all components of the residual vectors towards zero. The normalization step ensures all zones contribute equally to the penalty, regardless of their absolute temperature values. This prevents zones with naturally higher temperatures from dominating the regularization term.

#### 3.5 PUTTING TOGETHER THE NEURAL NETWORK OBJECTIVE

We now discuss the design of our final neural network. We formulate the objective in such a way that we can plug the above proposed regularizers in a standalone neural network architecture trained to regress output temperatures given a set of easily available input entities at each time step of a furnace operation. While starting the furnace operation, ambient temperatures are readily available (depicting the *initial state of the furnace*), along with walk interval, desired target set point temperatures. Then, based on the firing rates chosen for the burners of the furnace, there would be a resulting flow pattern in the furnace. This is a result of heat flow, and mass flow within the furnace (mass flow happens because of the slab movements, which need to be heated). This flow pattern would cause a change in the overall enthalpy, leading to a new temperature profile (*new state*) of the furnace, which can be measured by the resulting new gas and surface zone temperatures. These temperatures in turn could serve as input temperatures for the next step's prediction. For a more intuitive understanding of furnace operation, please refer Section [A.8.](#page-23-0)

In a practical setup, a neural network deployed could expect to consume the previous step temperatures, firing rates, walk interval, and set point temperatures as inputs. The output could then be the new

**329**

**334**

**354 355 356**

**358 359**

**361**

**364**

**369**

temperatures, and the next firing rates as well. With input-output data X ={(x (i) , y (i) )} N <sup>i</sup>=1 acquired in this manner, we can estimate parameters θ of a neural network fθ(.) by training it to predict y (i) given x (i) , for all time step i, as:

$$\theta^* \leftarrow \arg \min_{\theta} \mathcal{L}_{sup} \quad (11)$$

Here, Lsup = <sup>E</sup>(x(i),y(i))∈X [||y (i) − fθ(x (i) )||<sup>2</sup> 2 ] is a standard *supervised term for regression*. To make such a network physics-aware, all we need to do is include the above proposed terms Lebv and Lebs into the final objective. It should be noted that, in doing so, we do not need to make any architectural changes to the network in terms of inputs and outputs. Also, all auxiliary variables used in computation of [\(8\)](#page-4-1) and [\(9\)](#page-5-0) are only used during training of a physics-aware network, and are not required in the inference.

The regularization terms are computed using additional vectors as described earlier, influence the learning because they have the temperature terms in them. For example, in [\(10\)](#page-5-1), v<sup>g</sup> depends on gas zone temperatures Tg,j via g(g)arr, gleave in [\(8\)](#page-4-1). While computing Lebv we obtain the Tg,j terms using the network output, which are associated with the computational graph and thus help the updates during back-propagation. On the other hand, s(g)arr is associated with Ts,j which are detached for back-propagation while updating gas zone temperatures.

Similarly, in [\(10\)](#page-5-1), v<sup>s</sup> depends on surface zone temperatures Ts,j via s(s)arr, sleave in [\(9\)](#page-5-0). While computing Lebs we obtain the Ts,j terms using the network output, which are associated with the computational graph and thus help the updates during back-propagation. On the other hand, g(s)arr is associated with Tg,j which are detached for back-propagation while updating surface zone temperatures.

The overall physics-aware loss is formulated as:

$$\mathcal{L}_{total} = \mathcal{L}_{sup} + \lambda_{ebv} \mathcal{L}_{ebv} + \lambda_{ebs} \mathcal{L}_{ebs} \quad (12)$$

When calculating the physics-aware loss terms we detach certain temperature terms associated with one zone type (e.g., surface zone temperatures) during updates of the other zone type (e.g., gas zone temperatures). This prevents the network from altering these relationships unnaturally during backpropagation. As analogy, we can refer to a Teacher-Student Learning setup: Imagine the network learning from a teacher (the EB equations) that provides the correct temperature relationships. Detaching specific terms allows the network to focus on learning the mapping between furnace inputs and its own predicted zone temperatures, while still adhering to the guidance provided by the teacher (the EB equations) through the physics-aware loss terms. Algorithm [1](#page-6-0) provides detailed steps of our proposed approach.

Algorithm 1 Algorithm of the proposed method

1: Input: X ={(x (i) , y (i) )} N <sup>i</sup>=1, furnace configuration (set points and walk interval). maxeps > 0. 2: Initialize θ, TEAs, λebv, λebs > 0. 3: Initialize t<sup>G</sup> ∈ <sup>R</sup> |G| , t<sup>S</sup> ∈ <sup>R</sup> <sup>|</sup>S<sup>|</sup> with ambient temperatures, and firing rates. 4: for EN=1 to maxeps do ▷ EN: Epoch No. 5: for i=1 to N do ▷ i: time step 6: Compute DFAs ↼ GG(t) , ↼ GS(t) , ↼ SG(t) , ↼ SS(t) using [\(6\)](#page-4-2) and [\(7\)](#page-4-3). 7: Compute Lebv using [\(8\)](#page-4-1) and [\(10\)](#page-5-1). 8: Compute Lebs using [\(9\)](#page-5-0) and [\(10\)](#page-5-1). 9: Compute Lsup using X . 10: θ (i) ← θ (i−1) − η∇θLtotal ▷ Using [\(12\)](#page-6-1) 11: end for 12: end for 13: θ <sup>∗</sup> ← θ N.maxeps 14: return θ ∗

# 4 EXPERIMENTS

In this section we report results on 11 datasets obtained using different configurations of a real-world furnace based on [Hu et al.](#page-11-2) [\(2019\)](#page-11-2) (details in Section [A.8.3\)](#page-31-0). Major objective of the experiments is

**381**

**384**

**386**

Table 1: Comparison of proposed methods on the N1-2 Dataset

|      | Dataset |        |        |       | N1-2  |       |      |        | 965   | 1220 1250 | 750  |       |       |         |
|------|---------|--------|--------|-------|-------|-------|------|--------|-------|-----------|------|-------|-------|---------|
|      | Metric/ |        | Method |       | MLP   | PBMLP | LSTM | PBLSTM | DLSTM | PBDLSTM   | KAN  | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG     | ( ↓    | )     | 113.4 | 35.6  | 33.0 | 26.7   | 117.1 | 32.4      | 24.3 | 22.6  | 130.6 | 29.3    |
| RMSE |         | tS fur |        | ( ↓ ) | 116.4 | 22.4  | 25.6 | 11.7   | 114.4 | 24.9      | 15.2 | 14.6  | 119.1 | 20.4    |
| RMSE | tS      |        | obs    | ( ↓ ) | 106.9 | 43.4  | 61.1 | 66.5   | 109.3 | 67.4      | 35.1 | 33.6  | 139.8 | 45.4    |
|      | MAE     | tG     | ( ↓    | )     | 89.5  | 28.2  | 27.4 | 16.9   | 100.9 | 27.2      | 21.4 | 19.9  | 129.1 | 26.8    |
| MAE  | tS      | fur    | (      | ↓ )   | 96.2  | 17.8  | 21.5 | 9.9    | 101.1 | 20.1      | 14.3 | 13.8  | 118.6 | 19.5    |
| MAE  | tS      | obs    | (      | ↓ )   | 79.9  | 29.6  | 39.4 | 31.4   | 86.9  | 44.4      | 29.8 | 29.3  | 136.3 | 39.8    |
|      | mMAPE   | fr     | (      | ↓ )   | 176.6 | 58.5  | 29.5 | 23.5   | 201.0 | 26.2      | 44.2 | 32.6  | 200.8 | 27.8    |

Table 2: Comparison of proposed methods on the N2-1 Dataset

|      | Dataset |        |        |       | N2-1  |       |      |        | 955   | 1190 1250 | 750  |       |       |         |
|------|---------|--------|--------|-------|-------|-------|------|--------|-------|-----------|------|-------|-------|---------|
|      | Metric/ |        | Method |       | MLP   | PBMLP | LSTM | PBLSTM | DLSTM | PBDLSTM   | KAN  | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG     | ( ↓    | )     | 121.1 | 45.4  | 36.8 | 37.0   | 123.4 | 28.3      | 29.5 | 18.0  | 95.5  | 33.0    |
| RMSE |         | tS fur |        | ( ↓ ) | 123.8 | 27.6  | 29.5 | 28.9   | 120.5 | 18.7      | 20.7 | 8.8   | 80.6  | 24.9    |
| RMSE | tS      |        | obs    | ( ↓ ) | 113.1 | 52.4  | 65.6 | 63.3   | 114.5 | 51.9      | 41.0 | 27.2  | 90.7  | 51.7    |
|      | MAE     | tG     | ( ↓    | )     | 96.9  | 38.8  | 31.3 | 31.4   | 106.9 | 19.7      | 26.2 | 15.4  | 93.5  | 30.3    |
| MAE  | tS      | fur    | (      | ↓ )   | 103.6 | 24.8  | 26.7 | 25.5   | 106.4 | 16.5      | 19.8 | 7.7   | 80.1  | 24.1    |
| MAE  | tS      | obs    | (      | ↓ )   | 87.4  | 39.9  | 46.2 | 44.2   | 92.2  | 21.9      | 35.9 | 22.9  | 86.6  | 46.5    |
|      | mMAPE   | fr     | (      | ↓ )   | 187.6 | 67.8  | 28.4 | 29.8   | 210.6 | 24.9      | 43.7 | 34.2  | 212.3 | 26.2    |

to consider different neural network architectures with and without our proposed regularizers (and keeping everything else constant). Any gains reported could be attributed to our proposed regularizers that seek to enhance the physics-awareness of a network. Results across all the 11 datasets are reported in Tables [6,](#page-19-0) [7,](#page-20-0) [8,](#page-20-1) [9.](#page-21-0)

For neural network architectures, we study following variants: MLP, LSTM, a stacked/deep LSTM (DLSTM) and recently proposed KAN and xLSTM. We use commonly used regression performance metrics such as RMSE and MAE for the temperature prediction. We also report MAPE additionally for predicting the next firing rates (MAPE is more suitable due to the range of values that firing rates take). A metric against each of the different entities has been reported. For example, RMSE tS fur denotes the average RMSE for all the furnace surface zone predictions, RMSE tS obs denotes the average RMSE for all the obstacle surface zone predictions, RMSE tG denotes the average RMSE for all the gas zone predictions. mMAPE fr indicates the performance on the firing rate predictions. For all metrics, a lower value indicates a better performance. All metrics are reported along the rows of a table, and the columns represent the different methods. For each row, the best performing metric corresponding to a method is shown in bold.

In Table [1](#page-7-0) we report the performance of the architectures MLP, LSTM, DLSTM, KAN and xLSTM on the N1-2 dataset. We also report performances of PBMLP, PBLSTM, PBDLSTM, PBKAN and PBxLSTM, which are the Physics-Based (PB) variants of MLP, LSTM, DLSTM, KAN and PBxLSTM respectively. The green colored cells indicate that a PB variant has obtained a better performance than a vanilla variant without our proposed regularizers. Compared to the simpler MLP, we could see massive gains by the PBMLP.

The DLSTM (and xLSTM) variant possibly tends to overfit due to stacking of more LSTM layers, and performs worse compared to a vanilla LSTM model. Stacking LSTMs offered no advantage likely due to the data's inherent structure. Unlike language tasks that benefit from complex LSTM modeling with longer windows/time steps, zone-based method only requires capturing the relationship between the current state (s(i)) and the next (s(i+1)). Our data generation (details in Appendix) captures the relationship between current state (s(i)) and next state (s(i+1)), making complex LSTM architectures unnecessary. Initial experiments confirmed this, showing no significant improvement with longer windows compared to the simpler s(i), s(i+1) pairs. This aligns with Occam's razor - favoring simpler models with comparable performance.

However, when equipped with our regularizers, the PBDLSTM (and PBxLSTM) method obtains much better performance than the DLSTM (and xLSTM). The vanilla LSTM which performs better than the MLP and DLSTM, also obtains improvements after using the physics based regularizers, as indicated by the performance of PBLSTM. We also notice KAN to perform better than the base MLP (as observed in recent literature). In fact, the PBKAN variant performs the best among all methods at times.

In Table [2](#page-7-1) we report performances of the same approaches on the N2-1 dataset. We observed similar conclusions: the PB variants were outperforming their vanilla variants (as shown by green), thus depicting the benefit of the proposed regularizers. In this case, we observed that the PBKAN method obtains the best performance among all.

Table 3: Comparison of proposed methods on average across the datasets.

![](_page_8_Figure_1.jpeg)

Figure 3: Plot of actual (blue) and predicted (red) temperatures (in ◦C) across all obstacle surface zones using PBMLP. In (a) we omit previous furnace temperatures from the neural network input to show that performance degrades.

Difference in the datasets N1-2 and N2-1 comes by varying setpoint temperatures of the first and second control zones of the furnace. This shows that depending on the furnace configuration of the same geometry, the performance of a deep learning model may vary as the data distribution changes due to the difference in underlying physical entities. However, if equipped with physics based regularizers, we could make the network adhere to the governing laws, and get a reasonable predictive performance.

We further report on how the different methods perform across varying configurations or datasets on average, in Table [3.](#page-8-0) We observed similar performances, where the PB variants led to better performance. In Tables [6,](#page-19-0) [7,](#page-20-0) [8,](#page-20-1) [9](#page-21-0) we report the performances of the compared approaches across all the 11 datasets. We noticed that not only the PB variants obtain a better performance throughout, they are also more stable across different datasets as indicated by their standard deviations.

In Figure [4](#page-13-11) we plot the convergence of our PBMLP method. Losses with respect to all the individual terms converge well. In Figure [3](#page-8-1) we report visual plots of actual and predicted temperatures for PBMLP. We also show that omitting previous temperatures from the neural network inputs leads to an worse performance, thus, highlighting the impact of a furnace state on the model performance. We conducted a sensitivity analysis of λebv and λebs in Figure [5,](#page-14-0) observing stable performance across values.

#### 4.1 FINAL NOTE ON IMPACT OF ENERGY-BALANCE REGULARIZATION

Throughout the text, for all baseline methods in a column, the counterpart with the PB- prefix (eg, PBMLP, PBLSTM, PBDLSTM, PBKAN, PBxLSTM) indicates the usage of energy-balance regularization terms, and the green colored metrics all denote the consistent performance boost, as compared to the vanilla variants (eg, MLP, LSTM, DLSTM, KAN, xLSTM).

#### 4.2 COMPARISON AGAINST RECENT STATE-OF-THE-ART (SOTA)

While we acknowledge the importance of contextualizing our work, we recognize that making direct comparisons is challenging due to the unique characteristics of our framework. Most existing methods in the literature focus on limited exchange areas in furnace temperature modeling. In contrast, our robust data generation framework encompasses the entire set of exchange areas, which is essential for accurate temperature profiling.

To facilitate meaningful comparisons, we relate our results to established baselines recognized as State-Of-The-Art (SOTA) techniques in settings similar to ours. Specifically, we evaluate the impact of our research by comparing our proposed Physics-Based (PB) variants against the following methods: i) MLRVPST [\(Bao et al.](#page-9-0) [\(2023\)](#page-9-0)) and ii) PTDL-LSTM [\(de Souza Lima et al.](#page-10-6) [\(2023\)](#page-10-6)), the

**504**

**506**

**509**

**514 515 516**

**518 519**

**524**

**529**

**539**

Table 4: Comparison of proposed methods on average across the datasets against recent SOTA.

|      | Dataset Metric/ |     | Method |       | MLRVPST (Bao et al. (2023) ) | PTDL-LSTM (de Souza Lima et al. | Average (2023) ) PBLSTM | PBDLSTM | PBKAN | PBxLSTM |
|------|-----------------|-----|--------|-------|------------------------------|---------------------------------|-------------------------|---------|-------|---------|
|      | RMSE            | tG  | ( ↓    | )     | 31.2                         | 37.2                            | 30.4                    | 27.9    | 19.3  | 31.7    |
| RMSE |                 | tS  | fur    | ( ↓ ) | 24.5                         | 27.1                            | 20.2                    | 20.5    | 12.4  | 24.2    |
| RMSE | tS              |     | obs    | ( ↓ ) | 51.1                         | 64.9                            | 64.1                    | 61.7    | 29.8  | 45.8    |
|      | MAE             | tG  | ( ↓    | )     | 28.8                         | 29.7                            | 23.8                    | 22.4    | 16.8  | 29.5    |
| MAE  | tS              | fur | (      | ↓ )   | 23.7                         | 23.1                            | 18.1                    | 17.3    | 11.6  | 23.5    |
| MAE  | tS              | obs | (      | ↓ )   | 45.9                         | 40.7                            | 38.6                    | 36.0    | 25.7  | 40.5    |
|      | mMAPE           | fr  | (      | ↓ )   | 29.6                         | 39.2                            | 26.7                    | 25.9    | 39.3  | 37.5    |

latter of which is comparable to our LSTM implementation. The results of the comparisons are presented in Table [4.](#page-9-5) We observed that our proposed variants outperform the SOTA in general. The full set of results are presented in Tables [11,](#page-21-1) [12,](#page-22-0) [13,](#page-22-1) and [14.](#page-23-1)

# 5 CONCLUSIONS

This work proposes a novel regularization technique that leverages the Hottel Zone method to make deep neural networks *physics-aware* for improved furnace temperature profile prediction. Our approach is effective across various network architectures, including Multi-Layer Perceptrons (MLPs), Long Short-Term Memory (LSTM) networks, Kolmogorov-Arnold Networks (KANs) and Extended LSTM (xLSTM), as evidenced on datasets based on real-world furnace configurations with varying set points. In Sections [A.9](#page-32-0) and [A.10,](#page-32-1) we respectively discuss further real-life applications of our work, along with limitations of our work and future research directions.

#### ACKNOWLEDGMENTS

The authors wish to acknowledge

# ETHICS STATEMENT

There are no ethical concerns related to our work.

# REPRODUCIBILITY STATEMENT

Sections [A.4,](#page-18-0) [A.6,](#page-20-2) [A.8.2,](#page-27-0) and [A.8.3](#page-31-0) respectively aim at ensuring reproducibility at the following four levels: 1. Architectural and training details (e.g. number of epochs, hyper-parameters used, etc), 2. PyTorch-styled code for understanding of the implementation, 3. Algorithmic methodology used to generate dataset for ML model training, and 4. Exact data set creations and splits used for training and evaluation, with details.

# REFERENCES


[1] Yunqi Ban, Xianpeng Wang, Guodong Zhao, and Jian Wu. *Multiobjective Operation Optimization of*

[2] *Reheating Furnace based on Data Analytics*, 2023. [2,](#page-1-0) [17](#page-16-1) Qingfeng Bao, Sen Zhang, Jin Guo, Zhiqiang Li, and Zhenquan Zhang. Multivariate linear-regression variable parameter spatio-temporal zoning model for temperature prediction in steel rolling reheating furnace. *Journal of Process Control*, 123:108–122, 2023. [2,](#page-1-0) [9,](#page-8-2) [10,](#page-9-6) [18,](#page-17-0) [21](#page-20-3) Laura Boca de Giuli. *Physics-based neural network modelling, predictive control and lifelong learning applied to district heating systems*, 2023. [2,](#page-1-0) [18](#page-17-0) Felix Bunning, Benjamin Huber, Adrian Schalbetter, Ahmed Aboudonia, Mathias Hudoba de Badyn, ¨ Philipp Heer, Roy S Smith, and John Lygeros. Physics-informed linear regression is competitive with two machine learning methods in residential building mpc. *Applied Energy*, 310:118491, 2022. [2,](#page-1-0) [18](#page-17-0) Shengze Cai, Zhicheng Wang, Sifan Wang, Paris Perdikaris, and George Em Karniadakis. Physicsinformed neural networks for heat transfer problems. *Journal of Heat Transfer*, 143(6):060801, 2021. [2,](#page-1-0) [18](#page-17-0)

[3] **554 555 556**

[4] **559**

[5] **561**

[6] **564**

[7] **569**

[8] **579**

[9] **584**

[10] Chien-Jung Chen, Fu-I Chou, and Jyh-Horng Chou. Temperature prediction for reheating furnace by gated recurrent unit approach. *IEEE Access*, 10:33362–33369, 2022. [2,](#page-1-0) [18](#page-17-0) M De Beer, CG Du Toit, and PG Rousseau. A methodology to investigate the contribution of conduction and radiation heat transfer to the effective thermal conductivity of packed graphite pebble beds, including the wall effect. *Nuclear Engineering and Design*, 314:67–81, 2017. [16](#page-15-0) Rodrigo de Souza Lima, Leonardo Azevedo Scardua, and Gustavo Maia de Almeida. Predicting ´ temperatures inside a steel slab reheating furnace using deep learning. *Seven Editora*, 2023. [2,](#page-1-0) [9,](#page-8-2) [10,](#page-9-6) [17,](#page-16-1) [21](#page-20-3) Jan Drgo ´ na, Aaron R Tuor, Vikas Chandan, and Draguna L Vrabie. Physics-constrained deep learning ˇ of multi-zone building thermal dynamics. *Energy and Buildings*, 243:110992, 2021. [2,](#page-1-0) [18](#page-17-0) Hadi Ebrahimi, Akbar Zamaniyan, Jafar S Soltan Mohammadzadeh, and Ali Asghar Khalili. Zonal modeling of radiative heat transfer in industrial furnaces using simplified model for exchange area calculation. *Applied Mathematical Modelling*, 37(16-17):8004–8015, 2013. [2,](#page-1-0) [17](#page-16-1) Heather N Emady, Kellie V Anderson, William G Borghard, Fernando J Muzzio, Benjamin J Glasser, and Alberto Cuitino. Prediction of conductive heating time scales of particles in a rotary drum. *Chemical Engineering Science*, 152:45–54, 2016. [16](#page-15-0) EPSRC report. EPSRC report. [https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/V026402/1) [GrantRef=EP/V026402/1](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/V026402/1), 2020. [1,](#page-0-0) [15](#page-14-1) YT Feng and K Han. An accurate evaluation of geometric view factors for modelling radiative heat transfer in randomly packed beds of equally sized spheres. *International journal of heat and mass transfer*, 55:6374–6383, 2012. URL [https://doi.org/10.1016/j.](https://doi.org/10.1016/j.ijheatmasstransfer.2012.06.025) [ijheatmasstransfer.2012.06.025](https://doi.org/10.1016/j.ijheatmasstransfer.2012.06.025). [2](#page-1-0) SL Costa Ferreira, RE Bruns, Hadla Sousa Ferreira, Geraldo Domingues Matos, JM David, GC Brandao, EG Paranhos da Silva, LA Portugal, PS Dos Reis, AS Souza, et al. Box-behnken ˜ design: an alternative for the optimization of analytical methods. *Analytica chimica acta*, 597(2): 179–186, 2007. [29](#page-28-0) Juan Jose Garc ´ ´ıa-Esteban, Jorge Bravo-Abad, and Juan Carlos Cuevas. Deep learning for the modeling and inverse design of radiative heat transfer. *Physical Review Applied*, 16(6):064006, 2021. [2,](#page-1-0) [17](#page-16-1) Daniel Halme Stahlberg. Digital twin of a reheating furnace, 2021. ˚ [2,](#page-1-0) [17](#page-16-1) Jiawei Han, Mehrdad Mesgarpour, Lazarus Godson Asirvatham, Somchai Wongwises, Ho Seon Ahn, and Omid Mahian. A hyper-optimisation method based on a physics-informed machine learning and point clouds for a flat plate solar collector. *Journal of Thermal Analysis and Calorimetry*, pp. 1–20, 2023. [2,](#page-1-0) [18](#page-17-0) Zhili He, Futao Ni, Weiguo Wang, and Jian Zhang. A physics-informed deep learning method for solving direct and inverse heat conduction problems of materials. *Materials Today Communications*, 28:102719, 2021. [2,](#page-1-0) [18](#page-17-0) HC Hottel and ES Cohen. Radiant heat exchange in a gas-filled enclosure: Allowance for nonuniformity of gas temperature. *AIChE Journal*, 4(1):3–14, 1958. [16,](#page-15-0) [29](#page-28-0) Hoyt C Hottel and Adel F Saforim. *Radiative transfer*. McGraw-Hill, 1967. [16,](#page-15-0) [29](#page-28-0) Yukun Hu, CK Tan, Jonathan Broughton, and Paul Alun Roach. Development of a first-principles hybrid model for large-scale reheating furnaces. *Applied Energy*, 173:555–566, 2016. [1,](#page-0-0) [4,](#page-3-4) [5,](#page-4-4) [16,](#page-15-0) [28,](#page-27-1) [29,](#page-28-0) [31](#page-30-0) Yukun Hu, CK Tan, Jonathan Broughton, Paul Alun Roach, and Liz Varga. Model-based multiobjective optimisation of reheating furnace operations using genetic algorithm. *Energy Procedia*, 142:2143–2151, 2017. [2,](#page-1-0) [17](#page-16-1)

[11] **604**

[12] **606**

[13] **614 615**

[14] **617**

[15] **619**

[16] **629**

[17] **634**

[18] **636**

[19] Yukun Hu, CK Tan, Jonathan Broughton, Paul Alun Roach, and Liz Varga. Nonlinear dynamic simulation and control of large-scale reheating furnace operations using a zone method based model. *Applied Thermal Engineering*, 135:41–53, 2018. [2,](#page-1-0) [17](#page-16-1) Yukun Hu, CK Tan, John Niska, Jahedul Islam Chowdhury, Nazmiye Balta-Ozkan, Liz Varga, Paul Alun Roach, and Chunsheng Wang. Modelling and simulation of steel reheating processes under oxy-fuel combustion conditions–technical and environmental perspectives. *Energy*, 185: 730–743, 2019. [1,](#page-0-0) [7,](#page-6-2) [15,](#page-14-1) [25,](#page-24-0) [26,](#page-25-0) [28](#page-27-1) Soonsung Hwang, Gunwoo Jeon, Jongpil Jeong, and JunYoul Lee. A novel time series based seq2seq model for temperature prediction in firing furnace process. *Procedia Computer Science*, 155: 19–26, 2019. [2,](#page-1-0) [18](#page-17-0) IOM3 report. IOM3 report. [https://www.iom3.org/resource/](https://www.iom3.org/resource/transforming-foundations-industries.html) [transforming-foundations-industries.html](https://www.iom3.org/resource/transforming-foundations-industries.html), 2023. [1,](#page-0-0) [15](#page-14-1) Jung Hyun Jang, Dong Eun Lee, Man Young Kim, and Hyong Gon Kim. Investigation of the slab heating characteristics in a reheating furnace with the formation and growth of scale on the slab surface. *International Journal of Heat and Mass Transfer*, 53(19-20):4326–4332, 2010. [2,](#page-1-0) [17](#page-16-1) Gang Jing, Chenguang Ning, Jingwen Qin, Xudong Ding, Peiyong Duan, Haitao Liu, and Huiyun Sang. Physics-guided framework of neural network for fast full-field temperature prediction of indoor environment. *Journal of Building Engineering*, 68:106054, 2023. [2,](#page-1-0) [18](#page-17-0) George Em Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. *Nature Reviews Physics*, 3(6):422–440, 2021. [2,](#page-1-0) [18](#page-17-0) Jong Gyu Kim and Kang Y Huh. Prediction of transient slab temperature distribution in the re-heating furnace of a walking-beam type for rolling of steel slabs. *ISIJ international*, 40(11):1115–1123, 2000. [2,](#page-1-0) [17](#page-16-1) Kyung Mo Kim, Paul Hurley, and Juliana Pacheco Duarte. Physics-informed machine learning-aided framework for prediction of minimum film boiling temperature. *International Journal of Heat and Mass Transfer*, 191:122839, 2022. [2,](#page-1-0) [18](#page-17-0) Man Young Kim. A heat transfer model for the analysis of transient heating of the slab in a direct-fired walking beam type reheating furnace. *International Journal of Heat and Mass Transfer*, 50(19-20): 3740–3748, 2007. [2,](#page-1-0) [17](#page-16-1) Manu Lahariya, Farzaneh Karami, Chris Develder, and Guillaume Crevecoeur. Physics-informed lstm network for flexibility identification in evaporative cooling system. *IEEE Transactions on Industrial Informatics*, 19(2):1484–1494, 2022. [2,](#page-1-0) [18](#page-17-0) Guojun Li, Wenchao Ji, Linyang Wei, and Zhi Yi. A novel fuel supplies scheme based on the retrieval solutions of the decoupled zone method for reheating furnace. *International Communications in Heat and Mass Transfer*, 141:106572, 2023. [2,](#page-1-0) [17](#page-16-1) Kang Li. Eng-genes: a new genetic modelling approach for nonlinear dynamic systems. *IFAC Proceedings Volumes*, 38(1):162–167, 2005. [2,](#page-1-0) [17](#page-16-1) Tian Liang, Shanshan Wang, Chunyang Lu, Nan Jiang, Wenqi Long, Min Zhang, and Ruiqin Zhang. Environmental impact evaluation of an iron and steel plant in china: Normalized data and direct/indirect contribution. *Journal of Cleaner Production*, 264:121697, 2020. [1,](#page-0-0) [15](#page-14-1) Ying-Xin Liao, Jin-Hua She, and Min Wu. Integrated hybrid-pso and fuzzy-nn decoupling control for temperature of reheating furnace. *IEEE transactions on industrial electronics*, 56(7):2704–2714, 2009. [2,](#page-1-0) [17](#page-16-1) Jan Marti, Andreas Haselbacher, and Aldo Steinfeld. A numerical investigation of gas-particle suspensions as heat transfer media for high-temperature concentrated solar power. *International Journal of Heat and Mass Transfer*, 90:1056–1070, 2015. [2,](#page-1-0) [16](#page-15-0)

[20] **648 649 654 656 659 661 664 665 669 674 678 679 680 681 682 684 686 689 690 691 694 695 696 697 698 699 700** AD Matthew, CK Tan, PA Roach, J Ward, J Broughton, and A Heeley. Calculation of the radiative heat-exchange areas in a large-scale furnace with the use of the monte carlo method. *Journal of Engineering Physics and Thermophysics*, 87(3):732–742, 2014. [3,](#page-2-0) [29](#page-28-0) Matthieu Melot, Jean-Yves Trepanier, Ricardo Camarero, and Eddy Petro. Comparison of two ´ models for radiative heat transfer in high temperature thermal plasmas. *Modelling and Simulation in Engineering*, 2011, 2011. [2,](#page-1-0) [17](#page-16-1) Christopher L Muhich, Brian D Ehrhart, Ibraheam Al-Shankiti, Barbara J Ward, Charles B Musgrave, and Alan W Weimer. A review and perspective of efficient hydrogen generation via solar thermal water splitting. *Wiley Interdisciplinary Reviews: Energy and Environment*, 5(3):261–287, 2016. [2](#page-1-0) Net Zero by 2050. Net zero by 2050: A roadmap for the global energy sector. [https://www.iea.](https://www.iea.org/reports/net-zero-by-2050) [org/reports/net-zero-by-2050](https://www.iea.org/reports/net-zero-by-2050), 2021. [15](#page-14-1) Xuan Manh Nguyen, Pedro Rodriguez-Ayerbe, F Lawayeb, Didier Dumur, and Alain Mouchette. Temperature control of reheating furnace based on distributed model predictive control. In *2014 18th International Conference on System Theory, Control and Computing (ICSTCC)*, pp. 726–731. IEEE, 2014. [2,](#page-1-0) [17](#page-16-1) Tobias Oschmann and Harald Kruggel-Emden. A novel method for the calculation of particle heat conduction and resolved 3d wall heat transfer for the cfd/dem approach. *Powder Technology*, 338: 289–303, 2018. [16](#page-15-0) Junho Park. *Hybrid Machine Learning and Physics-Based Modeling Approaches for Process Control and Optimization*. PhD thesis, Brigham Young University, 2022. [2,](#page-1-0) [18](#page-17-0) Wei Qin, Zilong Zhuang, Yang Liu, and Jie Xu. Sustainable service oriented equipment maintenance management of steel enterprises using a two-stage optimization approach. *Robotics and Computer-Integrated Manufacturing*, 75:102311, 2022. [1,](#page-0-0) [15](#page-14-1) Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational physics*, 378:686–707, 2019. [2,](#page-1-0) [18](#page-17-0) Ling Shen, Zhipeng Chen, Xinyi Wang, and Jianjun He. Soft sensor modeling for 3d transient temperature field of large-scale aluminum alloy workpieces based on multi-loss consistency optimization pinn. *Sensors*, 23(14):6371, 2023. [2,](#page-1-0) [18](#page-17-0) Guangwu Tang, Bin Wu, Dengqi Bai, Yufeng Wang, Rick Bodnar, and Chenn Q Zhou. Modeling of the slab heating process in a walking beam reheating furnace for process optimization. *International Journal of Heat and Mass Transfer*, 113:1142–1151, 2017. [2,](#page-1-0) [17](#page-16-1) Josef Tausendschon and Stefan Radl. Deep neural network-based heat radiation modelling between ¨ particles and between walls and particles. *International Journal of Heat and Mass Transfer*, 177: 121557, 2021. [2,](#page-1-0) [17](#page-16-1) Hong-Chuong Tran and Yu-Lung Lo. Heat transfer simulations of selective laser melting process based on volumetric heat source with powder size consideration. *Journal of Materials Processing Technology*, 255:411–425, 2018. [2](#page-1-0) Ruihang Wang, Zhiwei Cao, Xin Zhou, Yonggang Wen, and Rui Tan. Phyllis: Physics-informed lifelong reinforcement learning for data center cooling control. In *Proceedings of the 14th ACM International Conference on Future Energy Systems*, pp. 114–126, 2023. [2,](#page-1-0) [18](#page-17-0) Gregor D Wehinger. Radiation matters in fixed-bed cfd simulations. *Chemie Ingenieur Technik*, 91 (5):583–591, 2019. [16](#page-15-0) Mark D Wilkinson, Michel Dumontier, IJsbrand Jan Aalbersberg, Gabrielle Appleton, Myles Axton, Arie Baak, Niklas Blomberg, Jan-Willem Boiten, Luiz Bonino da Silva Santos, Philip E Bourne, et al. The fair guiding principles for scientific data management and stewardship. *Scientific data*, 3 (1):1–9, 2016. [33](#page-32-2)

[21] **704**

[22] **706**

[23] **709**

[24] **721**

[25] **724**

[26] **729 730**

[27] **754**

[28] Hong Yu, Jiangnan Gong, Guoyin Wang, and Xiaofang Chen. A hybrid model for billet tapping temperature prediction and optimization in reheating furnace. *IEEE Transactions on Industrial Informatics*, 2022. [2,](#page-1-0) [17](#page-16-1) Qing-bo Yu, Zhong-wu Lu, and Jiu-ju Cai. Calculating method for influence of material flow on energy consumption in steel manufacturing process. *Journal of Iron and Steel Research, International*, 14(2):46–51, 2007. [1,](#page-0-0) [15](#page-14-1) Walter W Yuen. Rad-nnet, a neural network based correlation developed for a realistic simulation of the non-gray radiative heat transfer effect in three-dimensional gas-particle mixtures. *International Journal of Heat and Mass Transfer*, 52(13-14):3159–3168, 2009. [2,](#page-1-0) [17](#page-16-1) Walter W Yuen and Ezra E Takara. The zonal method: A practical solution method for radiative transfer in nonisothermal inhomogeneous media. *Annual review of heat transfer*, 8, 1997. [1,](#page-0-0) [3,](#page-2-0) [14,](#page-13-12) [17,](#page-16-1) [28,](#page-27-1) [29](#page-28-0) Silvia Maria Zanoli, Crescenzo Pepe, and Lorenzo Orlietti. Multi-mode model predictive control approach for steel billets reheating furnaces. *Sensors*, 23(8):3966, 2023. [2,](#page-1-0) [17](#page-16-1) Naiju Zhai and Xiaofeng Zhou. Temperature prediction of heating furnace based on deep transfer learning. *Sensors*, 20(17):4676, 2020. [2,](#page-1-0) [17](#page-16-1) Naiju Zhai, Xiaofeng Zhou, Shuai Li, and Haibo Shi. Soft sensor model for billet temperature in multiple heating furnaces based on transfer learning. *IEEE Transactions on Instrumentation and Measurement*, 2023. [2,](#page-1-0) [17](#page-16-1) Qi Zhang, Jin Xu, Yujie Wang, Ali Hasanbeigi, Wei Zhang, Hongyou Lu, and Marlene Arens. Comprehensive assessment of energy conservation and co2 emissions mitigation in china's iron and steel industry based on dynamic material flows. *Applied Energy*, 209:251–265, 2018. URL <https://doi.org/10.1016/j.apenergy.2017.10.084>. [1,](#page-0-0) [15](#page-14-1) Xingang Zhao, Koroush Shirvan, Robert K Salko, and Fengdi Guo. On the prediction of critical heat flux using a physics-informed machine learning-aided framework. *Applied Thermal Engineering*, 164:114540, 2020. [2,](#page-1-0) [18](#page-17-0) Jianhua Zhou, Yuwen Zhang, and JK Chen. Numerical simulation of laser irradiation to a randomly packed bimodal powder bed. *International Journal of Heat and Mass Transfer*, 52(13-14):3137– 3146, 2009. [2](#page-1-0)
# A APPENDIX

![](_page_13_Figure_2.jpeg)

Figure 4: Convergence of PBMLP in training, considering: a) Supervised, b) EBV, and c) EBS terms.

### A.1 MOTIVATION OF OUR WORK

[Yuen & Takara](#page-13-2) [\(1997\)](#page-13-2) in their study, have proved the elegance and superiority of the zone method over contemporary counterparts to model the physical phenomenon in high-temperature processes. In our work, we use the zone method towards a real-world application for the Foundation Industries (FIs), applied to reheating furnaces, due to the close and natural association/ relation of the zone-method

**759**

**761**

**764**

**766**

**769**

**779 780 781**

**784**

**804 805 806**

![](_page_14_Figure_0.jpeg)

Figure 5: Performance metrics against varying λebv = λebs = λ in PBMLP.

with the latter. Foundation Industries (FIs) constitute glass, metals, cement, ceramics, bulk chemicals, paper, steel, etc. and provide crucial, foundational materials for a diverse set of economically relevant industries: automobiles, machinery, construction, household appliances, chemicals, etc. FIs are heavy revenue and employment drivers, for instance, FIs in the United Kingdom (UK) economy are worth £52B [\(EPSRC report\)](#page-10-0), employ 0.25 million people, and comprise over 7000 businesses [\(IOM3](#page-11-0) [report\)](#page-11-0). The rapid acceleration in urbanization and industrialization over the decades has also led to improved building design and construction techniques. Great emphasis has been gradually placed on efficient heat generation, distribution, reduction, and optimized material usage.

However, despite their economic significance, as depicted by the above statistics, the FIs leverage energy-intensive methods. This makes FIs major industrial polluters and the largest consumers of natural resources across the globe. For example, in the UK, they produce 28 million tonnes of materials per year, and generate 10% of the entire UK's CO<sup>2</sup> emissions [\(EPSRC report;](#page-10-0) [IOM3](#page-11-0) [report\)](#page-11-0). Similarly, in China, the steel industry accounted for 15% of the total energy consumption, and 15.4% of the total CO<sup>2</sup> emissions [\(Zhang et al.,](#page-13-0) [2018;](#page-13-0) [Liang et al.,](#page-11-1) [2020\)](#page-11-1). These numbers put a challenge for the FIs in meeting our commitment to reduce net Green-House Gas (GHG) emissions, globally.

Various approaches have been relied upon to achieve the Net-Zero trajectory in FIs [\(Net Zero by](#page-12-12) [2050\)](#page-12-12): switching of grids to low carbon alternatives via green electricity, sustainable bio-fuel, and hydrogen sources, Carbon Capture and Storage (CCS), material reuse and recycling, etc. However, among all transformation enablers, a more proactive way to address the current challenges would be to tackle the core issue of process efficiency, via digitization, computer-integrated manufacturing, and control systems. Areas of impact by digitization could be reducing plant downtime, material and energy savings, resource efficiency, and industrial symbiosis, to name a few. Various computer-aided studies have already been conducted in notable industrial scenarios. The NSG Group's Pilkington UK Limited explored a sensor-driven Machine Learning (ML) model for product quality variation prediction (up to 72h), to reduce CO<sup>2</sup> emission by 30% till 2030 [\(IOM3 report\)](#page-11-0). Similar studies on service-oriented enterprise solutions for the steel industry have also been done recently in China [\(Qin](#page-12-0) [et al.,](#page-12-0) [2022\)](#page-12-0).

In this work, we tackle the key challenge of accurate and real-time temperature prediction in reheating furnaces, which are the energy-intensive bottlenecks common across the FIs. To give a perspective to the reader on why this is important, considering any process industry, such as the steel industry, one can observe that at the core, lies the process of conversion of materials (e.g., iron) into final products. This is done using a series of unit processes [\(Yu et al.,](#page-13-1) [2007\)](#page-13-1). The production process involves key steps such as dressing, sintering, smelting, casting, rolling, etc. A nice illustration of the different stages and processes in the steel industry can be found in [Qin et al.](#page-12-0) [\(2022\)](#page-12-0). The equipment in such process industries operates in high-intensity environments (e.g., high temperature), and has bottleneck components such as reheating furnaces, which require complex restart processes post-failure. This causes additional labor costs and energy consumption. Thus, for sustainable manufacturing, it is important to monitor the operating status of the furnaces via the furnace temperature profile.

A few studies [\(Hu et al.,](#page-11-2) [2019\)](#page-11-2) have shown promise in achieving notable fuel consumption reduction by reducing the overall heating time by even as less as 13 minutes while employing alternate combustion fuels. A key area of improvement for furnace operating status monitoring lies in leveraging efficient computational temperature control mechanisms within them. This is because energy consumption per kilogram of CO<sup>2</sup> could be reduced by a reduction in overall heating time.

As existing computational surrogate models have predictive capability bottlenecks, DL approaches can be used as suitable alternatives for real-time prediction. However, as only a handful of sensors/ thermo-couples could be physically placed within real-world furnaces (and that too at specific furnace

**814 815**

**817**

**819**

**829**

**834**

**836**

**854**

**856**

walls), the challenge of obtaining good-quality real-world data at scale to train DL models in such scenarios remains infeasible. To alleviate this, we identify the classical Hottel's zone method [\(Hottel](#page-10-12) [& Cohen,](#page-10-12) [1958;](#page-10-12) [Hottel & Saforim,](#page-10-13) [1967\)](#page-10-13) which provides an elegant, iterative way to computationally model the temperature profile within a furnace, requiring only a few initial entities which are easily measurable. However, straightforward utilization of the same is not suitable for real-time deployment and prediction, due to computational expensiveness. For this reason, we propose that we generate an offline data set using the zone method, consisting of input-output pairs to train and evaluate ML models. We will provide a detailed description of the data generation methodology using the zone method.

# A.1.1 COMPUTATIONAL MODELS

Available computational surrogate models based on Computational Fluid Dynamics (CFD) [\(Wehinger,](#page-12-13) [2019;](#page-12-13) [De Beer et al.,](#page-10-14) [2017\)](#page-10-14), Discrete Element Method (DEM) [\(Emady et al.,](#page-10-15) [2016\)](#page-10-15), CFD-DEM hybrids [\(Oschmann & Kruggel-Emden,](#page-12-14) [2018\)](#page-12-14), Two Fluid Models (TFM) [\(Marti et al.,](#page-11-3) [2015\)](#page-11-3), etc. incur expensive and time-consuming data acquisition, design, optimization, and high inference times. To break through the predictive capability bottlenecks of these surrogate models, DL approaches can be suitable candidates for real-time prediction, owing to their accuracy and inherently faster inference times (often only in the order of milliseconds).

# A.1.2 DISCUSSION ON COMPUTATIONAL ASPECTS

In general, PINNs/ PCNNs and accurate simulators (e.g., CFD models) are two different approaches to solving a physical problem. In terms of computational efficiency, they cannot be compared at the same level. While PCNNs could take milliseconds for inference, accurate simulators have difficulty even achieving real-time simulation. Thus, PCNNs have the potential to be integrated directly into a control system for real-time control. This is because PCNNs are a type of approaches that encode the governing equations of the problem into the network training, whereas, accurate simulators are based on numerical methods that discretize the problem domain and solve the equations on a mesh, which can be time-consuming, and challenging to generate for complex geometries or moving boundaries (such as the furnace studied in our work).

Generally speaking, the zone method is faster and simpler to implement than the CFD method. For example, even with a consumer-level PC, to simulate a 341-min real reheating process, the zone model only takes 5 mins, but CFD models often take several days, if not weeks, to provide *useful* results [\(Hu et al.,](#page-10-1) [2016\)](#page-10-1). Therefore, in this study, we utilize the zone model to generate training data for PCNNs. In future studies, the trained PCNNs will be integrated directly into furnace control systems. For our study, typically, generating 1500 timesteps of data for a single furnace using the zone method took about 2 hours, including the time for setting different configurations.

However, talking about the absolute time of a CFD case simulation itself depends on many factors, such as mesh density, sub-model selection, step size settings, and computer hardware configuration. Specific to our case, using the same configuration of PC, CFD simulation of the steady-state operating conditions of each setting takes about 5 hours. So the total time taken is 5 hours multiplied by the number of simulated working conditions. For the simulation of unsteady operating conditions, CFD is currently very difficult to implement, and some simplifications must be made. The specific time consumption depends on the duration of the simulated unsteady process. For the real process of 341 min for the case we studied, CFD would take at least 5 days (vs, 5 min of the zone method). As for the neural-network based implementations, for ML-based inference on a Apple M2 Max 32GB, our PCNN takes roughly 0.5s for inferring the entire furnace profile for a single time step instance, given the input variables as discussed.

#### A.1.3 COMPUTATIONAL EFFICIENCY (TRAINING AND TESTING TIME) BETWEEN METHODS WITH AND WITHOUT ENERGY-BALANCE BASED PHYSICS-REGULARIZATION

The training time per mini-batch/iteration increases by up to 10x for smaller batch sizes when compared to the vanilla variant without Energy-Balance (EB) regularization. This increase is primarily due to the various matrix multiplications involving the DFA/TEA terms with higher-order matrices, particularly from the surface zones that comprise the regularization terms. However, when considering absolute run times, the increase is minimal; for example, the runtime per mini-batch is

**869**

**874**

**884**

**886**

**889 890 891**

**904**

**906**

**909**

**917**

approximately 76.11 seconds/iteration. We could reduce this further by using larger batch sizes to fully leverage GPU capabilities, although the performance gains would be marginal. In contrast, the simpler vanilla variants have a runtime of about 7.48 seconds/iteration.

During inference, the time remains the same for both variants, as the regularization terms are only required during training for the Physics-Based (PB) variants, with no changes in the architectures.

#### A.2 DETAILS OF RELATED WORK

While the research conducted in this work is at nascent stage, we believe it could pave way for further developments from an ML perspective, to solve a real-world application problem with value in terms of environmental sustainability. Our work, for an applied physical sciences reader, could inspire how ML and DL could be used to address a niche domain scenario. At the same time, for an ML audience, we believe that our work showcases a novel way to integrate physics based constraints into a neural network, especially using the zone method. Arguably, there exists a plethora of works related to PINNs, however, using PINNs to incorporate the zone method based regularizers as in our work, is a novel contribution to the community. The motivation to leverage the zone method also comes from the fact that it provides an elegant (and superior) way, as studied by [Yuen & Takara](#page-13-2) [\(1997\)](#page-13-2), to model the physical phenomenon in high-temperature processes inside reheating furnaces.

In this section, we exhaustively present a set of relevant approaches with which our work can be loosely associated with. Specifically, we categorize them into two major classes: i) nonlinear dynamic systems, radiative heat transfer and view factor modeling, and, ii) modeling in reheating furnaces. We also talk about PINNs, and how our method is unique with respect to the existing literature.

(Category 1) Nonlinear dynamic systems, radiative heat transfer and view factor modeling: Our work at its heart is based on the zone method, which in turn relies on notions of radiative heat transfer and view factor modeling (or interchangeably, exchange area calculation). Describing the behavior of a furnace state involves combustion models, control loops, set point calculations, and fuel flux control in zones. It also involves linearization and model order reduction for state estimation and state-space control. The inherent complexity makes the modeling a nonlinear dynamic system.

While there is no exact similarity, our work shares some common philosophies with few earlier works. For instance, [Ebrahimi et al.](#page-10-3) [\(2013\)](#page-10-3) discuss the modeling of radiative heat transfer using simplified exchange area calculation. Radiative heat transfer in high-temperature thermal plasmas has been studied by [Melot et al.](#page-12-3) [\(2011\)](#page-12-3) while comparing two models. A nonlinear dynamic simulation and control based method has been studied by [Hu et al.](#page-11-4) [\(2018\)](#page-11-4). A classical work based on genetic algorithm for nonlinear dynamic systems [\(Li,](#page-11-5) [2005\)](#page-11-5) is also present, which, instead of a data-driven approach, leverages a pre-defined set of mathematical functions.

Within this category, some approaches have also employed neural networks. In [Yuen](#page-13-5) [\(2009\)](#page-13-5), a network was trained for simulating non-gray radiative heat transfer effect in 3D gas-particle mixtures. Some approaches have used networks for view factor modeling with DEM-based simulations [\(Tausendschon¨](#page-12-4) [& Radl,](#page-12-4) [2021\)](#page-12-4), and some have addressed the near-field heat transfer or close regime [\(Garc´ıa-Esteban](#page-10-4) [et al.,](#page-10-4) [2021\)](#page-10-4).

(Category 2) Modeling in reheating furnaces: We now discuss methods dealing with some form of prediction or optimization in reheating furnaces. Classically, [Kim & Huh](#page-11-8) [\(2000\)](#page-11-8) discussed a method to predict transient slab temperatures in a walking-beam furnace for rolling of steel slabs. [Kim](#page-11-9) [\(2007\)](#page-11-9) proposed a model for analyzing transient slab heating in a direct-fired walking beam furnace. [Jang](#page-11-10) [et al.](#page-11-10) [\(2010\)](#page-11-10) investigated the slab heating characteristics with the formation and growth of scale. [Tang](#page-12-5) [et al.](#page-12-5) [\(2017\)](#page-12-5) studied slab heating for process optimization. A distributed model predictive control approach was proposed in [Nguyen et al.](#page-12-6) [\(2014\)](#page-12-6). Few multi-objective optimization methods were discussed in [Hu et al.](#page-10-8) [\(2017\)](#page-10-8); [Ban et al.](#page-9-1) [\(2023\)](#page-9-1). A fuel supplies scheme based approach was proposed in [Li et al.](#page-11-11) [\(2023\)](#page-11-11). Other related works involved multi-mode model predictive control approach for steel billets [\(Zanoli et al.,](#page-13-8) [2023\)](#page-13-8), and a hybrid model for billet tapping temperature prediction [\(Yu](#page-13-9) [et al.,](#page-13-9) [2022\)](#page-13-9).

Some neural network based approaches in this category studied transfer learning [\(Zhai & Zhou,](#page-13-6) [2020;](#page-13-6) [Zhai et al.,](#page-13-7) [2023\)](#page-13-7), digital twin modeling [\(Halme Stahlberg](#page-10-5) ˚ , [2021\)](#page-10-5), and steel slab temperature prediction [\(de Souza Lima et al.,](#page-10-6) [2023\)](#page-10-6). [Liao et al.](#page-11-6) [\(2009\)](#page-11-6) discussed an integrated hybrid-PSO and fuzzy-NN decoupling based solution. Other works have studied aspects related to time-series

**924**

**929**

**954**

**956**

**959**

**961**

modeling [\(Hwang et al.,](#page-11-7) [2019;](#page-11-7) [Chen et al.,](#page-10-7) [2022\)](#page-10-7), and multivariate linear-regression in steel rolling [\(Bao et al.,](#page-9-0) [2023\)](#page-9-0).

PINNs: The methods mentioned above discuss alternatives aimed at modeling either exchange factors with radiative heat transfer, or specific slab temperature predictions in reheating furnaces. However, they do not explicitly address physics-based prior incorporation within their optimization frameworks, especially for the neural network variants. To this end, we now discuss a few relevant works in the body of literature on PINNs. For a detailed review on PINNs in general, we refer the interested reader to the papers by [Raissi et al.](#page-12-7) [\(2019\)](#page-12-7); [Karniadakis et al.](#page-11-12) [\(2021\)](#page-11-12). It should be noted that PINNs are a broad category of approaches, and the literature is vast. Here, we discuss those methods which relate to certain aspects of thermal modeling.

[Drgona et al.](#page-10-9) ˇ [\(2021\)](#page-10-9) proposed a physics-constrained method to model multi-zone building thermal dynamics. A multi-loss consistency optimization PINN [\(Shen et al.,](#page-12-8) [2023\)](#page-12-8) was proposed for largescale aluminium alloy workpieces. Other approaches focus on prototype heat transfer problems and power electronics applications [Cai et al.](#page-9-2) [\(2021\)](#page-9-2), minimum film boiling temperature [\(Kim et al.,](#page-11-13) [2022\)](#page-11-13), critical heat flux [\(Zhao et al.,](#page-13-10) [2020\)](#page-13-10), solving direct and inverse heat conduction problems of materials [\(He et al.,](#page-10-10) [2021\)](#page-10-10), lifelong learning in district heating systems [\(Boca de Giuli,](#page-9-3) [2023\)](#page-9-3), PINN and point clouds for flat plate solar collector [\(Han et al.,](#page-10-11) [2023\)](#page-10-11), residential building MPC [\(Bunning et al.](#page-9-4) ¨ , [2022\)](#page-9-4), hybrid ML and PINN for Process Control and Optimization [\(Park,](#page-12-9) [2022\)](#page-12-9), reinforcement learning for data center cooling control [\(Wang et al.,](#page-12-10) [2023\)](#page-12-10), flexibility identification in evaporative cooling [\(Lahariya et al.,](#page-11-14) [2022\)](#page-11-14), and fast full-field temperature prediction of indoor environment [\(Jing et al.,](#page-11-15) [2023\)](#page-11-15).

Uniqueness of our work within existing literature: While we have observed a number of loosely related methods as discussed above, upon a clear look at them, we can conclude the following:

- 1. Comparison with category 1 methods: Among the approaches focusing on view factor modeling with radiative transfer, the area of interest is often simplified. The modeling covers select few exchange areas. The methods are also geometry-specific. Our approach on the other hand seeks a generic, geometry-agnostic modeling that covers the entire set of exchange areas. The exchange areas can be intuitively perceived as those interfaces from where radiation can transfer, between a pair of zones (surface/gas). A background on exchange areas is provided in the proposed work section. The ones involving neural networks, often employ feed-forward Multi-Layer Perceptron (MLP) models with few hidden layers. As showcased in our experiments, a simple MLP trained to regress the outputs given certain inputs may not generalize well to unseen distributions, due to lack of explicit understanding of the underlying physics. On the other hand, we empirically showcase that our proposed PCNN performs better than such a baseline MLP. Within a single PCNN framework, our method can also cover other architectures such as LSTMs, KANs, xLSTMs etc.
- 2. Comparison with category 2 methods: Both non-neural and neural-network based methods presented in this category, as observed, focus on predicting temperatures only in certain regions of a furnace, often, the slab temperature profiling. Our work, on the other hand aims at achieving a complete furnace temperature profiling, ranging from the gas zones, to both types of surface zones: furnace walls as well as the slab/obstacle surfaces. Our training data set is obtained based on the iterative zone method, and is more holistic in nature as compared to the discussed methods. This makes an apple-to-apple comparison difficult with other methods as they deal with different problem setups. Furthermore, the neural methods in this category are not trained to be physics aware.
- 3. Comparison with PINNs: It should be noted that any PINN approach is driven by the priors corresponding to the underlying physical phenomenon. As we did not find PINN methods addressing zone method based modeling, we could claim our PCNN variant to be novel in nature, especially, in this studied problem setup. Essentially, casting the temperature prediction task in reheating furnaces as in our work, and modeling via explicit physicsconstrained regularizers (based on zone method) as done in our work, is a first of its kind. It is a simple paradigm, and could be used to build further sophisticated developments. At the same time, it simply requires input-output pairs (as shown later) to train the underlying ML/PCNN model, and makes no geometry-specific assumptions of the furnace. The data creation method discussed in our method is holistic, covers all possible exchange areas, and thus, is unique in nature itself.

**979**

**989 990 991**

**994**

**1011**

**1014 1015**

**1017**

#### A.3 PERFORMANCE METRICS

For a data set containing N samples: X = {(x (i) , y (i) )} N <sup>i</sup>=1, we make use of the following standard regression performance evaluation metrics:

1. Root Mean Squared Error (RMSE), defined as:

$$RMSE = \sqrt{\frac{\sum_{i=1}^N (\mathbf{y}^{(i)} - f_{\theta}(\mathbf{x}^{(i)}))^2}{N}} \quad (13)$$

2. Mean Absolute Error (MAE), defined as:

$$MAE = \frac{\sum_{i=1}^N |\mathbf{y}^{(i)} - f_\theta(\mathbf{x}^{(i)})|}{N} \quad (14)$$

Mean Absolute Percentage Error (MAPE) is unsuitable for firing rate prediction due to potential division by zero. We use a modified MAPE (mMAPE) with a small epsilon (ϵ = 0.05) added to the denominator:

$$mMAPE = \frac{1}{N} \sum_{t=1}^N \left| \frac{f_t - \hat{f}_t}{f_t + \epsilon} \right| \quad (15)$$

Here, f<sup>t</sup> is the actual firing rate, and ˆf<sup>t</sup> is the predicted value.

We evaluate model performance for each entity (gas zone temperatures, tG; furnace surface temperatures, tS fur; obstacle surface temperatures, tS obs; firing rates, fr) separately as: RMSE tG, RMSE tS fur, RMSE tS obs, MAE tG, MAE tS fur, MAE tS obs, and mMAPE fr. Performance metrics (RMSE, MAE, mMAPE) are computed using corresponding predictions from the model (fθ(x (i) )) and ground truth values from the data (y (i) ). Results are presented for the test split (standard practice). mMAPE is evaluated only for the firing rates. RMSE, MAE and mMAPE range in [0, ∞] with lower values indicating better performance (↓) as shown in the tables.

#### A.4 TRAINING DETAILS AND MODEL ARCHITECTURES

We train our PBMLP for 10 epochs using PyTorch (early stopping to avoid over-fitting), and report results with the final checkpoint. For the EB equations, we perform the same normalization for enthalpy, flux, and temperatures, as in the final neural network output as discussed earlier. We found a learning rate of 0.001 with Adam optimizer and batch size of 64 to be optimal, along with ReLU non-linearity.

We pick the [50,100,200] configuration for hidden layers, i.e., 3 hidden layers, with 50, 100, and 200 neurons respectively. We use λebv = λebs = 0.1. In general, a value lesser than 1 is observed to be better, otherwise, the model focuses less on the regression task. Following are values of other variables: |G| = 24, |S| = 178 (76 furnace surface zones and 102 obstacle surface zones), N<sup>g</sup> = 6, and Stefan-Boltzmann constant=5.6687e-08. Unless otherwise stated, this is the setting we use to report any results for our method, for example, while comparing with other methods. Please note that the MLP baseline has exactly the same training configuration as the PBMLP except that it does not use the physics regularizers.

We provide details about the LSTM variants used. The LSTM variant has a single LSTM layer with 50 hidden nodes, followed by FC layer-1 with 50 input nodes and 100 output nodes, FC layer-2 with 100 input nodes and 200 output nodes. Both FC layer-1 and FC layer-2 have ReLU non-linearity. Lastly, there is a final FC layer with sigmoid nonlinearity that maps to the number of output features as in the data set. The DLSTM variant has three stacked LSTM layers, each with 100 hidden nodes, followed by a final FC layer with sigmoid nonlinearity. As we can see, we have kept the total number of layers in LSTM and DSLTM comparable to that of the baseline MLP.

For the xLSTM implementation, we follow a similar architeture as the DLSTM model. Similar to the DLSTM we place a LSTM layer that maps the input to 100 hidden nodes. However, after that, instead of stacking two more LSTM layers, we place a single xLSTM block stack (as mentioned in the official repository <https://github.com/NX-AI/xlstm>). After the xLSTM block, the remaining layers are similar to that of the DLSTM. Within the xLSTM block stack, the sLSTM block has 4 heads,

**1029**

**1034**

**1054**

**1056**

**1071**

**1079**

conv1d kernel size=4, and, the mLSTM block has conv1d kernel size=4, qkv proj blocksize=4, and 4 heads. Overall, xLSTM block has context length of 1, 7 blocks, and embedding dimension of 100. For KAN, we follow the implementation suggestions as in [https://github.com/](https://github.com/KindXiaoming/pykan) [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan) and use a single hidden layer with one neuron. Interestingly, the KAN despite being simpler than the MLP baseline, is not only easier to train, but also outperforms the MLP, as evidenced in many contemporary works. Broadly speaking, the training specific hyperparameters across all the compared models are the same (e.g., number of epochs, optimizer, batch size, learning rate, etc). The only difference comes from their respective architectures. For a similar architecture, the additional difference for the physics based variants lie in terms of usage of the additional regularization terms. Table [5](#page-19-1) summarizes the details.

Table 5: Architectural and training details across different studied models

| Model           |        |         | Architecture |        | Layer-specific information                                                       |
|-----------------|--------|---------|--------------|--------|----------------------------------------------------------------------------------|
| MLP 3 hidden    |        | layers  | (50,         | 100,   | 200 neurons)+                                                                    |
| Final           | FC     | layer   | (no.         | of     | outputs)                                                                         |
| 1               | LSTM   | layer   | (50          | hidden | nodes) +                                                                         |
| 2               | FC     | layers  | (FC-1        | and    | FC-2) +                                                                          |
| Final           | FC     | layer   | (no.         | of     | outputs)                                                                         |
|                 |        |         |              |        | FC-1: 50-100, FC-2: 100-200                                                      |
| DLSTM 3 stacked | LSTM   | layers  | (100         |        | hidden nodes each)                                                               |
| +               | Final  | FC      | layer        | (no.   | of outputs)                                                                      |
| xLSTM 1 LSTM    |        | layer   | (100         | hidden | nodes) +                                                                         |
| 1 xLSTM         | block  | + Final | FC           |        | layer (no. of outputs)                                                           |
|                 |        |         |              |        | xLSTM block: context length = 1, #blocks =7, embedding dim = 100                 |
|                 |        |         |              |        | sLSTM block:#heads=4, conv1d kernel size=4                                       |
|                 |        |         |              |        | mLSTM block: #heads=4,                                                           |
|                 |        |         |              |        | conv1d kernel size=4, qkv proj blocksize=4                                       |
| KAN 1           | hidden |         | layer        | (1     | neuron)+                                                                         |
| Final           | FC     | layer   | (no.         | of     | outputs)                                                                         |
| PB-variants     |        | Same    | as           |        | corresponding base architecture, but additionally use physics-based regularizers |
|                 |        |         |              |        | with λ ebv = λ ebs = 0 1                                                         |
|                 | Common |         |              |        | Hyperparameters: 10 epochs, Adam optimizer, lr=0.001, batch size=64              |

Table 6: All results (Normal Type 1 Datasets)

|      |         |     |        |       |       |       | Table 6: | All results | (Normal | Type 1    | Datasets) |       |       |         |
|------|---------|-----|--------|-------|-------|-------|----------|-------------|---------|-----------|-----------|-------|-------|---------|
|      | Dataset |     |        |       | N1-1  |       |          |             | 925     | 1220 1250 | 750       |       |       |         |
|      | Metric/ |     | Method |       | MLP   | PBMLP | LSTM     | PBLSTM      | DLSTM   | PBDLSTM   | KAN       | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG  | ( ↓    | )     | 136.4 | 55.3  | 15.6     | 43.3        | 28.4    | 16.1      | 40.7      | 12.6  | 39.6  | 13.7    |
| RMSE |         | tS  | fur    | ( ↓ ) | 139.2 | 39.8  | 7.1      | 39.3        | 13.8    | 6.3       | 34.4      | 9.7   | 38.3  | 10.6    |
| RMSE | tS      |     | obs    | ( ↓ ) | 124.8 | 64.9  | 43.7     | 73.8        | 54.2    | 52.6      | 54.2      | 21.2  | 63.9  | 22.8    |
|      | MAE     | tG  | ( ↓    | )     | 108.6 | 51.0  | 11.1     | 39.5        | 20.7    | 10.9      | 38.8      | 10.2  | 37.5  | 11.7    |
| MAE  | tS      | fur | (      | ↓ )   | 115.7 | 39.2  | 6.0      | 38.1        | 12.2    | 5.1       | 34.1      | 9.1   | 37.8  | 10.0    |
| MAE  | tS      | obs | (      | ↓ )   | 100.2 | 54.8  | 19.5     | 58.1        | 32.1    | 22.1      | 50.1      | 18.1  | 59.3  | 18.7    |
|      | mMAPE   | fr  | (      | ↓ )   | 232.9 | 70.7  | 25.6     | 26.5        | 21.9    | 23.7      | 51.1      | 40.7  | 22.1  | 27.6    |
|      | Dataset |     |        |       | N1-2  |       |          |             | 965     | 1220 1250 | 750       |       |       |         |
|      | Metric/ |     | Method |       | MLP   | PBMLP | LSTM     | PBLSTM      | DLSTM   | PBDLSTM   | KAN       | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG  | ( ↓    | )     | 113.4 | 35.6  | 33.0     | 26.7        | 117.1   | 32.4      | 24.3      | 22.6  | 130.6 | 29.3    |
| RMSE |         | tS  | fur    | ( ↓ ) | 116.4 | 22.4  | 25.6     | 11.7        | 114.4   | 24.9      | 15.2      | 14.6  | 119.1 | 20.4    |
| RMSE | tS      |     | obs    | ( ↓ ) | 106.9 | 43.4  | 61.1     | 66.5        | 109.3   | 67.4      | 35.1      | 33.6  | 139.8 | 45.4    |
|      | MAE     | tG  | ( ↓    | )     | 89.5  | 28.2  | 27.4     | 16.9        | 100.9   | 27.2      | 21.4      | 19.9  | 129.1 | 26.8    |
| MAE  | tS      | fur | (      | ↓ )   | 96.2  | 17.8  | 21.5     | 9.9         | 101.1   | 20.1      | 14.3      | 13.8  | 118.6 | 19.5    |
| MAE  | tS      | obs | (      | ↓ )   | 79.9  | 29.6  | 39.4     | 31.4        | 86.9    | 44.4      | 29.8      | 29.3  | 136.3 | 39.8    |
|      | mMAPE   | fr  | (      | ↓ )   | 176.6 | 58.5  | 29.5     | 23.5        | 201.0   | 26.2      | 44.2      | 32.6  | 200.8 | 27.8    |
|      | Dataset |     |        |       | N1-3  |       |          |             | 995     | 1220 1250 | 750       |       |       |         |
|      | Metric/ |     | Method |       | MLP   | PBMLP | LSTM     | PBLSTM      | DLSTM   | PBDLSTM   | KAN       | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG  | ( ↓    | )     | 31.1  | 30.5  | 39.3     | 39.2        | 100.0   | 35.7      | 23.1      | 20.9  | 114.9 | 30.1    |
| RMSE |         | tS  | fur    | ( ↓ ) | 22.1  | 24.3  | 8.0      | 16.5        | 97.0    | 25.8      | 18.4      | 17.1  | 104.3 | 23.1    |
| RMSE | tS      |     | obs    | ( ↓ ) | 54.4  | 47.8  | 69.0     | 77.4        | 97.2    | 60.5      | 27.7      | 26.4  | 124.2 | 35.1    |
|      | MAE     | tG  | ( ↓    | )     | 23.0  | 23.8  | 25.3     | 29.1        | 87.0    | 29.4      | 20.9      | 18.4  | 113.6 | 27.9    |
| MAE  | tS      | fur | (      | ↓ )   | 16.8  | 20.8  | 6.4      | 14.6        | 85.8    | 22.4      | 17.7      | 16.4  | 104.1 | 22.4    |
| MAE  | tS      | obs | (      | ↓ )   | 31.4  | 29.4  | 36.6     | 46.5        | 73.1    | 32.7      | 24.0      | 22.5  | 120.7 | 30.4    |
|      | mMAPE   | fr  | (      | ↓ )   | 32.0  | 28.1  | 25.8     | 26.9        | 128.7   | 29.4      | 33.0      | 27.7  | 127.7 | 31.7    |

#### A.5 FULL SET OF RESULTS ON THE 11 DATASETS

In Tables [6,](#page-19-0) [7,](#page-20-0) [8,](#page-20-1) [9](#page-21-0) we report the performances of the compared approaches across all the 11 datasets. We noticed that not only the PB variants obtain a better performance throughout, they are also more stable across different datasets as indicated by their standard deviations (Table [10\)](#page-21-2). On the other hand, the performances of the vanilla networks were not stable across different datasets.

However, we also noted that Physics-Based (PB) variants perform *slightly worse* than the vanilla methods in certain datasets. This because we did not tune hyperparameters for each configuration, but rather aimed to obtain average performance across configurations. While there may be potential for further improvements at the configuration level, our primary goal was to assess the generalizability of our approach. In real-world scenarios, variability is to be expected. It is possible that, for certain

**1099**

**1104**

**1106**

**1109**

**1119**

Table 7: All results (Normal Type 2 Datasets)

|      | Dataset |     |        |       | N2-1  |       |      |        | 955   | 1190 1250 | 750  |       |       |         |
|------|---------|-----|--------|-------|-------|-------|------|--------|-------|-----------|------|-------|-------|---------|
|      | Metric/ |     | Method |       | MLP   | PBMLP | LSTM | PBLSTM | DLSTM | PBDLSTM   | KAN  | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG  | ( ↓    | )     | 121.1 | 45.4  | 36.8 | 37.0   | 123.4 | 28.3      | 29.5 | 18.0  | 95.5  | 33.0    |
| RMSE |         | tS  | fur    | ( ↓ ) | 123.8 | 27.6  | 29.5 | 28.9   | 120.5 | 18.7      | 20.7 | 8.8   | 80.6  | 24.9    |
| RMSE | tS      |     | obs    | ( ↓ ) | 113.1 | 52.4  | 65.6 | 63.3   | 114.5 | 51.9      | 41.0 | 27.2  | 90.7  | 51.7    |
|      | MAE     | tG  | ( ↓    | )     | 96.9  | 38.8  | 31.3 | 31.4   | 106.9 | 19.7      | 26.2 | 15.4  | 93.5  | 30.3    |
| MAE  | tS      | fur | (      | ↓ )   | 103.6 | 24.8  | 26.7 | 25.5   | 106.4 | 16.5      | 19.8 | 7.7   | 80.1  | 24.1    |
| MAE  | tS      | obs | (      | ↓ )   | 87.4  | 39.9  | 46.2 | 44.2   | 92.2  | 21.9      | 35.9 | 22.9  | 86.6  | 46.5    |
|      | mMAPE   | fr  | (      | ↓ )   | 187.6 | 67.8  | 28.4 | 29.8   | 210.6 | 24.9      | 43.7 | 34.2  | 212.3 | 26.2    |
|      | Dataset |     |        |       | N2-2  |       |      |        | 955   | 1230 1250 | 750  |       |       |         |
|      | Metric/ |     | Method |       | MLP   | PBMLP | LSTM | PBLSTM | DLSTM | PBDLSTM   | KAN  | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG  | ( ↓    | )     | 116.1 | 39.2  | 34.3 | 34.6   | 122.5 | 33.3      | 27.6 | 18.0  | 135.5 | 31.0    |
| RMSE |         | tS  | fur    | ( ↓ ) | 118.6 | 24.3  | 28.4 | 27.9   | 119.9 | 27.3      | 19.6 | 9.7   | 123.9 | 23.9    |
| RMSE | tS      |     | obs    | ( ↓ ) | 108.7 | 45.2  | 64.0 | 61.7   | 113.6 | 70.7      | 39.6 | 29.0  | 144.8 | 50.2    |
|      | MAE     | tG  | ( ↓    | )     | 91.1  | 32.9  | 29.5 | 29.7   | 105.4 | 28.9      | 24.7 | 15.5  | 134.0 | 28.7    |
| MAE  | tS      | fur | (      | ↓ )   | 96.7  | 20.8  | 25.8 | 24.6   | 105.8 | 23.9      | 18.8 | 8.8   | 123.3 | 23.2    |
| MAE  | tS      | obs | (      | ↓ )   | 82.8  | 32.5  | 44.4 | 42.5   | 91.2  | 49.6      | 34.4 | 24.6  | 141.3 | 44.9    |
|      | mMAPE   | fr  | (      | ↓ )   | 187.1 | 66.7  | 28.4 | 30.0   | 220.4 | 25.6      | 46.8 | 35.0  | 220.6 | 26.7    |

Table 8: All results (Normal Type 3 Datasets)

|      | Dataset |     |        |       | N3-1  |       |      |        | 955   | 1220 1250 | 750  |       |       |         |
|------|---------|-----|--------|-------|-------|-------|------|--------|-------|-----------|------|-------|-------|---------|
|      | Metric/ |     | Method |       | MLP   | PBMLP | LSTM | PBLSTM | DLSTM | PBDLSTM   | KAN  | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG  | ( ↓    | )     | 119.5 | 42.9  | 34.4 | 34.7   | 122.7 | 33.3      | 27.6 | 18.0  | 135.5 | 31.0    |
| RMSE |         | tS  | fur    | ( ↓ ) | 122.5 | 24.1  | 28.5 | 27.9   | 120.1 | 27.4      | 19.6 | 9.7   | 123.9 | 23.9    |
| RMSE | tS      |     | obs    | ( ↓ ) | 111.3 | 45.5  | 64.1 | 61.9   | 113.7 | 70.7      | 39.6 | 28.8  | 144.8 | 50.2    |
|      | MAE     | tG  | ( ↓    | )     | 94.6  | 36.6  | 29.6 | 29.7   | 105.5 | 28.9      | 24.7 | 15.5  | 134.1 | 28.7    |
| MAE  | tS      | fur | (      | ↓ )   | 101.5 | 20.3  | 25.8 | 24.7   | 105.9 | 24.0      | 18.8 | 8.7   | 123.3 | 23.2    |
| MAE  | tS      | obs | (      | ↓ )   | 85.1  | 33.3  | 44.4 | 42.6   | 91.3  | 49.6      | 34.4 | 24.5  | 141.3 | 44.9    |
|      | mMAPE   | fr  | (      | ↓ )   | 194.2 | 88.0  | 28.4 | 30.0   | 220.4 | 25.6      | 46.8 | 35.0  | 220.6 | 26.6    |
|      | Dataset |     |        |       | N3-2  |       |      |        | 955   | 1220 1280 | 750  |       |       |         |
|      | Metric/ |     | Method |       | MLP   | PBMLP | LSTM | PBLSTM | DLSTM | PBDLSTM   | KAN  | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG  | ( ↓    | )     | 23.8  | 17.9  | 19.5 | 19.5   | 17.3  | 18.1      | 14.9 | 14.5  | 16.4  | 15.9    |
| RMSE |         | tS  | fur    | ( ↓ ) | 11.2  | 7.8   | 12.0 | 11.2   | 9.6   | 10.5      | 6.8  | 7.3   | 9.4   | 9.2     |
| RMSE | tS      |     | obs    | ( ↓ ) | 57.6  | 41.6  | 54.5 | 52.0   | 61.9  | 61.6      | 26.0 | 26.7  | 33.9  | 34.8    |
|      | MAE     | tG  | ( ↓    | )     | 17.0  | 11.8  | 14.7 | 14.6   | 13.1  | 13.7      | 12.0 | 11.7  | 14.1  | 13.7    |
| MAE  | tS      | fur | (      | ↓ )   | 9.6   | 6.8   | 10.7 | 9.6    | 8.0   | 8.6       | 6.0  | 6.6   | 8.6   | 8.3     |
| MAE  | tS      | obs | (      | ↓ )   | 31.5  | 20.1  | 27.7 | 26.2   | 32.3  | 32.5      | 20.9 | 21.5  | 27.7  | 28.6    |
|      | mMAPE   | fr  | (      | ↓ )   | 37.5  | 41.9  | 25.2 | 27.2   | 22.1  | 22.9      | 51.2 | 50.6  | 21.5  | 22.9    |
|      | Dataset |     |        |       | N3-3  |       |      |        | 955   | 1220 1300 | 750  |       |       |         |
|      | Metric/ |     | Method |       | MLP   | PBMLP | LSTM | PBLSTM | DLSTM | PBDLSTM   | KAN  | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG  | ( ↓    | )     | 18.2  | 15.6  | 15.6 | 15.5   | 15.6  | 15.5      | 17.5 | 19.0  | 12.5  | 11.5    |
| RMSE |         | tS  | fur    | ( ↓ ) | 7.5   | 8.7   | 7.7  | 7.0    | 7.6   | 7.7       | 11.2 | 13.7  | 5.9   | 6.0     |
| RMSE | tS      |     | obs    | ( ↓ ) | 52.4  | 47.2  | 51.2 | 48.3   | 58.7  | 58.4      | 27.6 | 29.2  | 28.1  | 28.7    |
|      | MAE     | tG  | ( ↓    | )     | 11.0  | 11.7  | 10.2 | 10.2   | 11.3  | 11.2      | 15.2 | 17.1  | 10.7  | 10.0    |
| MAE  | tS      | fur | (      | ↓ )   | 6.0   | 7.1   | 6.0  | 5.4    | 6.4   | 6.4       | 10.6 | 13.0  | 5.4   | 5.3     |
| MAE  | tS      | obs | (      | ↓ )   | 23.4  | 24.4  | 22.2 | 21.1   | 26.1  | 26.3      | 23.2 | 24.8  | 22.5  | 22.9    |
|      | mMAPE   | fr  | (      | ↓ )   | 40.5  | 38.7  | 27.9 | 30.5   | 22.9  | 24.9      | 60.2 | 62.3  | 21.3  | 24.0    |

configurations, the underlying physics is better captured by a stronger vanilla architecture (e.g., LSTM vs. MLP). If the vanilla model is effectively learning and generalizing, the explicit regularization may yield minimal gains. However, we do not consider this a case of PB variants performing worse than vanilla methods; rather, their performance metrics are comparable.

Conversely, it is important to note that PB variants generally outperform vanilla variants by significant multiplicative factors in performance metrics.

The performances of the proposed Physics-Based (PB) approaches across all the 11 datasets are also compared against the following SOTA methods: i) MLRVPST [\(Bao et al.](#page-9-0) [\(2023\)](#page-9-0)) and ii) PTDL-LSTM [\(de Souza Lima et al.](#page-10-6) [\(2023\)](#page-10-6)), the results of which are presented in Tables [11,](#page-21-1) [12,](#page-22-0) [13,](#page-22-1) and [14.](#page-23-1) We notice that our proposed variants outperform the SOTA consistently in general.

### A.6 PSEUDO-CODES FOR OUR TRAINING FRAMEWORK

In Algorithm [2,](#page-24-1) we outline the key steps required in training our physics-constrained framework. The training involves a typical mini-batch based optimization, where each instance in a mini-batch contains the various entities obtained from one row/time step of the data set. The entities are present in their respective columns. The columns for the constant terms (e.g., (Q˙ conv)<sup>i</sup> , (Q˙ fuel,net)<sup>i</sup> , (Q˙ <sup>a</sup>)<sup>i</sup> , Ai( ˙qconv)<sup>i</sup> and Q˙ s,i) will have the values repeated across all the corresponding rows to create a dataloader.

**1154**

**1159**

**1171**

**1174 1175**

**1177**

Table 9: All results (Normal Type 4 Datasets)

|      | Dataset |     |        |       | N4-1  |       |       |        | 955   | 1220 1250 | 705  |       |       |         |
|------|---------|-----|--------|-------|-------|-------|-------|--------|-------|-----------|------|-------|-------|---------|
|      | Metric/ |     | Method |       | MLP   | PBMLP | LSTM  | PBLSTM | DLSTM | PBDLSTM   | KAN  | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG  | ( ↓    | )     | 117.4 | 39.3  | 110.8 | 34.2   | 29.6  | 31.5      | 27.1 | 17.3  | 93.3  | 92.9    |
| RMSE |         | tS  | fur    | ( ↓ ) | 121.9 | 32.9  | 98.2  | 30.2   | 19.7  | 26.3      | 20.6 | 8.6   | 80.1  | 79.1    |
| RMSE | tS      |     | obs    | ( ↓ ) | 115.7 | 64.3  | 126.2 | 67.3   | 48.7  | 53.4      | 47.0 | 23.1  | 94.6  | 94.7    |
|      | MAE     | tG  | ( ↓    | )     | 94.2  | 35.3  | 90.0  | 30.3   | 22.0  | 24.2      | 28.7 | 14.4  | 91.8  | 91.2    |
| MAE  | tS      | fur | (      | ↓ )   | 102.0 | 31.6  | 78.3  | 27.2   | 17.9  | 20.5      | 22.0 | 7.7   | 79.7  | 78.5    |
| MAE  | tS      | obs | (      | ↓ )   | 91.5  | 51.6  | 92.1  | 50.7   | 21.4  | 30.2      | 55.9 | 19.4  | 90.6  | 90.7    |
|      | mMAPE   | fr  | (      | ↓ )   | 123.0 | 19.9  | 141.7 | 21.6   | 22.3  | 28.0      | 22.4 | 17.2  | 139.9 | 141.0   |
|      | Dataset |     |        |       | N4-2  |       |       |        | 955   | 1220 1250 | 765  |       |       |         |
|      | Metric/ |     | Method |       | MLP   | PBMLP | LSTM  | PBLSTM | DLSTM | PBDLSTM   | KAN  | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG  | ( ↓    | )     | 38.7  | 36.1  | 34.2  | 24.2   | 121.9 | 32.4      | 27.3 | 18.0  | 135.5 | 30.5    |
| RMSE |         | tS  | fur    | ( ↓ ) | 27.0  | 23.2  | 27.9  | 13.4   | 119.3 | 26.6      | 19.3 | 10.2  | 123.8 | 23.5    |
| RMSE | tS      |     | obs    | ( ↓ ) | 63.9  | 44.4  | 61.9  | 65.7   | 111.4 | 69.2      | 37.3 | 31.2  | 142.6 | 47.9    |
|      | MAE     | tG  | ( ↓    | )     | 32.7  | 29.5  | 29.2  | 15.1   | 104.5 | 27.9      | 24.5 | 15.6  | 134.2 | 28.3    |
| MAE  | tS      | fur | (      | ↓ )   | 24.3  | 19.5  | 25.1  | 12.2   | 105.1 | 23.2      | 18.5 | 9.4   | 123.2 | 22.8    |
| MAE  | tS      | obs | (      | ↓ )   | 45.7  | 30.0  | 41.8  | 29.5   | 88.9  | 47.5      | 31.9 | 26.8  | 139.2 | 42.4    |
|      | mMAPE   | fr  | (      | ↓ )   | 42.9  | 59.7  | 30.2  | 23.3   | 229.6 | 25.7      | 49.8 | 37.0  | 230.2 | 27.6    |
|      | Dataset |     |        |       | N4-3  |       |       |        | 955   | 1220 1250 | 810  |       |       |         |
|      | Metric/ |     | Method |       | MLP   | PBMLP | LSTM  | PBLSTM | DLSTM | PBDLSTM   | KAN  | PBKAN | xLSTM | PBxLSTM |
|      | RMSE    | tG  | ( ↓    | )     | 35.5  | 28.0  | 35.3  | 25.2   | 120.3 | 30.2      | 27.0 | 33.4  | 27.6  | 29.4    |
| RMSE |         | tS  | fur    | ( ↓ ) | 21.8  | 19.3  | 25.5  | 8.7    | 117.1 | 23.8      | 18.1 | 27.4  | 20.9  | 21.9    |
| RMSE | tS      |     | obs    | ( ↓ ) | 46.1  | 48.0  | 53.2  | 67.5   | 105.7 | 62.8      | 31.7 | 51.8  | 40.6  | 42.1    |
|      | MAE     | tG  | ( ↓    | )     | 25.5  | 20.3  | 29.0  | 15.4   | 102.6 | 24.7      | 24.4 | 31.3  | 24.7  | 27.1    |
| MAE  | tS      | fur | (      | ↓ )   | 16.4  | 14.7  | 21.8  | 7.3    | 103.0 | 19.5      | 17.5 | 26.5  | 19.4  | 21.1    |
| MAE  | tS      | obs | (      | ↓ )   | 28.8  | 27.1  | 33.2  | 32.1   | 82.4  | 38.9      | 26.5 | 47.9  | 34.4  | 36.1    |
|      | mMAPE   | fr  | (      | ↓ )   | 57.5  | 50.0  | 40.3  | 24.6   | 259.6 | 28.2      | 61.0 | 60.0  | 28.0  | 30.6    |

Table 10: All results (standard deviations)

|      | Dataset Metric/ |     | Method |       | MLP  | PBMLP | LSTM | PBLSTM | DLSTM | STDEV PBDLSTM | KAN  | PBKAN | xLSTM | PBxLSTM |
|------|-----------------|-----|--------|-------|------|-------|------|--------|-------|---------------|------|-------|-------|---------|
|      | RMSE            | tG  | ( ↓    | )     | 48.2 | 11.6  | 25.9 | 8.7    | 48.8  | 7.5           | 6.7  | 5.4   | 51.1  | 21.8    |
| RMSE |                 | tS  | fur    | ( ↓ ) | 55.8 | 9.2   | 25.4 | 10.9   | 52.4  | 8.3           | 6.8  | 5.8   | 48.3  | 19.4    |
| RMSE | tS              |     | obs    | ( ↓ ) | 31.2 | 8.0   | 21.6 | 8.4    | 27.6  | 7.1           | 8.7  | 8.1   | 47.2  | 18.8    |
|      | MAE             | tG  | ( ↓    | )     | 39.3 | 11.8  | 21.5 | 9.5    | 43.3  | 7.3           | 7.0  | 5.5   | 51.5  | 21.9    |
| MAE  | tS              | fur | (      | ↓ )   | 46.4 | 9.5   | 20.2 | 10.4   | 46.2  | 7.2           | 7.0  | 5.8   | 48.5  | 19.5    |
| MAE  | tS              | obs | (      | ↓ )   | 30.0 | 10.8  | 19.3 | 11.3   | 30.2  | 10.6          | 11.0 | 8.0   | 48.2  | 19.1    |
|      | mMAPE           | fr  | (      | ↓ )   | 78.3 | 20.2  | 34.2 | 3.1    | 99.7  | 2.0           | 11.1 | 13.5  | 91.6  | 34.4    |

Table 11: All results against SOTA (Normal Type 1 Datasets)

|      | Dataset Metric/ |     | Method |       | MLRVPST | PTDL-LSTM | N1-1 PBLSTM | PBDLSTM | PBKAN | PBxLSTM |
|------|-----------------|-----|--------|-------|---------|-----------|-------------|---------|-------|---------|
|      | RMSE            | tG  | ( ↓    | )     | 45.4    | 15.6      | 43.3        | 16.1    | 12.6  | 13.7    |
| RMSE |                 | tS  | fur    | ( ↓ ) | 41.0    | 7.1       | 39.3        | 6.3     | 9.7   | 10.6    |
| RMSE | tS              |     | obs    | ( ↓ ) | 68.6    | 43.7      | 73.8        | 52.6    | 21.2  | 22.8    |
|      | MAE             | tG  | ( ↓    | )     | 43.2    | 11.1      | 39.5        | 10.9    | 10.2  | 11.7    |
| MAE  | tS              | fur | (      | ↓ )   | 40.5    | 6.0       | 38.1        | 5.1     | 9.1   | 10.0    |
| MAE  | tS              | obs | (      | ↓ )   | 64.2    | 19.5      | 58.1        | 22.1    | 18.1  | 18.7    |
|      | mMAPE           | fr  | (      | ↓ )   | 28.4    | 25.6      | 26.5        | 23.7    | 40.7  | 27.6    |
|      | Dataset         |     |        |       |         |           | N1-2        |         |       |         |
|      | Metric/         |     | Method |       | MLRVPST | PTDL-LSTM | PBLSTM      | PBDLSTM | PBKAN | PBxLSTM |
|      | RMSE            | tG  | ( ↓    | )     | 30.7    | 33.0      | 26.7        | 32.4    | 22.6  | 29.3    |
| RMSE |                 | tS  | fur    | ( ↓ ) | 22.1    | 25.6      | 11.7        | 24.9    | 14.6  | 20.4    |
| RMSE | tS              |     | obs    | ( ↓ ) | 48.8    | 61.1      | 66.5        | 67.4    | 33.6  | 45.4    |
|      | MAE             | tG  | ( ↓    | )     | 28.1    | 27.4      | 16.9        | 27.2    | 19.9  | 26.8    |
| MAE  | tS              | fur | (      | ↓ )   | 21.3    | 21.5      | 9.9         | 20.1    | 13.8  | 19.5    |
| MAE  | tS              | obs | (      | ↓ )   | 43.2    | 39.4      | 31.4        | 44.4    | 29.3  | 39.8    |
|      | mMAPE           | fr  | (      | ↓ )   | 31.8    | 29.5      | 23.5        | 26.2    | 32.6  | 27.8    |
|      | Dataset         |     |        |       |         |           | N1-3        |         |       |         |
|      | Metric/         |     | Method |       | MLRVPST | PTDL-LSTM | PBLSTM      | PBDLSTM | PBKAN | PBxLSTM |
|      | RMSE            | tG  | ( ↓    | )     | 27.8    | 39.3      | 39.2        | 35.7    | 20.9  | 30.1    |
| RMSE |                 | tS  | fur    | ( ↓ ) | 20.6    | 8.0       | 16.5        | 25.8    | 17.1  | 23.1    |
| RMSE | tS              |     | obs    | ( ↓ ) | 36.7    | 69.0      | 77.4        | 60.5    | 26.4  | 35.1    |
|      | MAE             | tG  | ( ↓    | )     | 25.1    | 25.3      | 29.1        | 29.4    | 18.4  | 27.9    |
| MAE  | tS              | fur | (      | ↓ )   | 19.4    | 6.4       | 14.6        | 22.4    | 16.4  | 22.4    |
| MAE  | tS              | obs | (      | ↓ )   | 31.5    | 36.6      | 46.5        | 32.7    | 22.5  | 30.4    |
|      | mMAPE           | fr  | (      | ↓ )   | 32.3    | 25.8      | 26.9        | 29.4    | 27.7  | 31.7    |

As observed in Algorithm [2,](#page-24-1) X train batch and y train batch correspond to x (i) and y (i) in X , and are used to compute tr loss regtmps representing Lsup in eq[\(12\)](#page-6-1). tr loss ebv and tr loss ebs respectively correspond to Lebv and Lebs in eq[\(12\)](#page-6-1). The collection of the T<sup>g</sup> terms for being associated with the computational graph for backpropagation by virtue of use in eq[\(8\)](#page-4-1), is done by y train pred[:,:n gas zones].

**1224**

**1227**

**1229**

Table 12: All results against SOTA (Normal Type 2 Datasets)

|      | Dataset Metric/ |     | Method |       | MLRVPST | PTDL-LSTM | N2-1 PBLSTM | PBDLSTM | PBKAN | PBxLSTM |
|------|-----------------|-----|--------|-------|---------|-----------|-------------|---------|-------|---------|
|      | RMSE            | tG  | ( ↓    | )     | 35.7    | 36.8      | 37.0        | 28.3    | 18.0  | 33.0    |
| RMSE |                 | tS  | fur    | ( ↓ ) | 27.8    | 29.5      | 28.9        | 18.7    | 8.8   | 24.9    |
| RMSE | tS              |     | obs    | ( ↓ ) | 55.5    | 65.6      | 63.3        | 51.9    | 27.2  | 51.7    |
|      | MAE             | tG  | ( ↓    | )     | 32.8    | 31.3      | 31.4        | 19.7    | 15.4  | 30.3    |
| MAE  | tS              | fur | (      | ↓ )   | 27.0    | 26.7      | 25.5        | 16.5    | 7.7   | 24.1    |
| MAE  | tS              | obs | (      | ↓ )   | 50.5    | 46.2      | 44.2        | 21.9    | 22.9  | 46.5    |
|      | mMAPE           | fr  | (      | ↓ )   | 30.6    | 28.4      | 29.8        | 24.9    | 34.2  | 26.2    |
|      | Dataset         |     |        |       |         |           | N2-2        |         |       |         |
|      | Metric/         |     | Method |       | MLRVPST | PTDL-LSTM | PBLSTM      | PBDLSTM | PBKAN | PBxLSTM |
|      | RMSE            | tG  | ( ↓    | )     | 33.4    | 34.3      | 34.6        | 33.3    | 18.0  | 31.0    |
| RMSE |                 | tS  | fur    | ( ↓ ) | 26.3    | 28.4      | 27.9        | 27.3    | 9.7   | 23.9    |
| RMSE | tS              |     | obs    | ( ↓ ) | 53.5    | 64.0      | 61.7        | 70.7    | 29.0  | 50.2    |
|      | MAE             | tG  | ( ↓    | )     | 30.8    | 29.5      | 29.7        | 28.9    | 15.5  | 28.7    |
| MAE  | tS              | fur | (      | ↓ )   | 25.5    | 25.8      | 24.6        | 23.9    | 8.8   | 23.2    |
| MAE  | tS              | obs | (      | ↓ )   | 48.3    | 44.4      | 42.5        | 49.6    | 24.6  | 44.9    |
|      | mMAPE           | fr  | (      | ↓ )   | 31.6    | 28.4      | 30.0        | 25.6    | 35.0  | 26.7    |

Table 13: All results against SOTA (Normal Type 3 Datasets)

Dataset N3-1

Metric/ Method MLRVPST PTDL-LSTM PBLSTM PBDLSTM PBKAN PBxLSTM RMSE tG (↓) 33.5 34.4 34.7 33.3 18.0 31.0 RMSE tS fur (↓) 26.5 28.5 27.9 27.4 9.7 23.9 RMSE tS obs (↓) 53.7 64.1 61.9 70.7 28.8 50.2 MAE tG (↓) 31.0 29.6 29.7 28.9 15.5 28.7 MAE tS fur (↓) 25.7 25.8 24.7 24.0 8.7 23.2 MAE tS obs (↓) 48.5 44.4 42.6 49.6 24.5 44.9 mMAPE fr (↓) 31.4 28.4 30.0 25.6 35.0 26.6

Dataset N3-2

Metric/ Method MLRVPST PTDL-LSTM PBLSTM PBDLSTM PBKAN PBxLSTM RMSE tG (↓) 18.0 19.5 19.5 18.1 14.5 15.9 RMSE tS fur (↓) 11.4 12.0 11.2 10.5 7.3 9.2 RMSE tS obs (↓) 38.1 54.5 52.0 61.6 26.7 34.8 MAE tG (↓) 15.7 14.7 14.6 13.7 11.7 13.7 MAE tS fur (↓) 10.7 10.7 9.6 8.6 6.6 8.3 MAE tS obs (↓) 32.0 27.7 26.2 32.5 21.5 28.6 mMAPE fr (↓) 27.2 25.2 27.2 22.9 50.6 22.9

Dataset N3-3

Metric/ Method MLRVPST PTDL-LSTM PBLSTM PBDLSTM PBKAN PBxLSTM RMSE tG (↓) 14.0 15.6 15.5 15.5 19.0 11.5 RMSE tS fur (↓) 8.2 7.7 7.0 7.7 13.7 6.0 RMSE tS obs (↓) 32.5 51.2 48.3 58.4 29.2 28.7 MAE tG (↓) 11.3 10.2 10.2 11.2 17.1 10.0 MAE tS fur (↓) 7.3 6.0 5.4 6.4 13.0 5.3 MAE tS obs (↓) 26.3 22.2 21.1 26.3 24.8 22.9 mMAPE fr (↓) 28.9 27.9 30.5 24.9 62.3 24.0

Similar role towards back-propagation via T<sup>s</sup> terms in eq[\(9\)](#page-5-0) is taken care of by y train pred[:,n gas zones:n gas zones+n fur surf zones+n obs surf zones].

get pb ebv pred() computes v<sup>g</sup> in eq[\(10\)](#page-5-1) for each instance (corresponding to a time-step of zone method) present in a mini-batch of the variables obtained from the already created data set. In doing so, each of the |G| elements of v<sup>g</sup> are computed using eq[\(8\)](#page-4-1) and the corresponding/relevant auxiliary variables from the data. sgarr plus hg tensor batch collects mini-batch terms using relevant terms like s(g)arr, h<sup>g</sup> in eq[\(10\)](#page-5-1) towards vg. The relevant DFA terms are collected in tensor dfa GG tensor batch. Similarly, we make use of get pb ebs pred(), dfa SS tensor batch, gsarr plus hs tensor batch for computing v<sup>s</sup> in eq[\(10\)](#page-5-1) and using eq[\(9\)](#page-5-0). Having obtained the dataset, it only involves sampling mini-batches via appropriate helper functions in any Deep Learning framework (e.g., PyTorch). In Algorithms [3](#page-25-1)[-4,](#page-26-0) we provide a few helper functions which can be useful to further understand the computation of some of the tensors involved in the training loop described in Algorithm [2.](#page-24-1)

# A.7 IN-DEPTH SENSITIVITY ANALYSIS OF PBMLP

We evaluated PBMLP's sensitivity to hyperparameters (loss terms, hidden layers, batch size, activation functions) using shuffled test data from all furnace configurations. To establish an upper bound on

**1267**

**1281**

**1284**

**1287**

Table 14: All results against SOTA (Normal Type 4 Datasets)

|      | Dataset Metric/ |     | Method |       | MLRVPST | PTDL-LSTM | N4-1 PBLSTM | PBDLSTM | PBKAN | PBxLSTM |
|------|-----------------|-----|--------|-------|---------|-----------|-------------|---------|-------|---------|
|      | RMSE            | tG  | ( ↓    | )     | 36.2    | 110.8     | 34.2        | 31.5    | 17.3  | 92.9    |
| RMSE |                 | tS  | fur    | ( ↓ ) | 30.9    | 98.2      | 30.2        | 26.3    | 8.6   | 79.1    |
| RMSE | tS              |     | obs    | ( ↓ ) | 62.5    | 126.2     | 67.3        | 53.4    | 23.1  | 94.7    |
|      | MAE             | tG  | ( ↓    | )     | 33.9    | 90.0      | 30.3        | 24.2    | 14.4  | 91.2    |
| MAE  | tS              | fur | (      | ↓ )   | 30.4    | 78.3      | 27.2        | 20.5    | 7.7   | 78.5    |
| MAE  | tS              | obs | (      | ↓ )   | 57.9    | 92.1      | 50.7        | 30.2    | 19.4  | 90.7    |
|      | mMAPE           | fr  | (      | ↓ )   | 20.2    | 141.7     | 21.6        | 28.0    | 17.2  | 141.0   |
|      | Dataset         |     |        |       |         |           | N4-2        |         |       |         |
|      | Metric/         |     | Method |       | MLRVPST | PTDL-LSTM | PBLSTM      | PBDLSTM | PBKAN | PBxLSTM |
|      | RMSE            | tG  | ( ↓    | )     | 32.2    | 34.2      | 24.2        | 32.4    | 18.0  | 30.5    |
| RMSE |                 | tS  | fur    | ( ↓ ) | 25.0    | 27.9      | 13.4        | 26.6    | 10.2  | 23.5    |
| RMSE | tS              |     | obs    | ( ↓ ) | 50.8    | 61.9      | 65.7        | 69.2    | 31.2  | 47.9    |
|      | MAE             | tG  | ( ↓    | )     | 29.7    | 29.2      | 15.1        | 27.9    | 15.6  | 28.3    |
| MAE  | tS              | fur | (      | ↓ )   | 24.2    | 25.1      | 12.2        | 23.2    | 9.4   | 22.8    |
| MAE  | tS              | obs | (      | ↓ )   | 45.3    | 41.8      | 29.5        | 47.5    | 26.8  | 42.4    |
|      | mMAPE           | fr  | (      | ↓ )   | 32.6    | 30.2      | 23.3        | 25.7    | 37.0  | 27.6    |
|      | Dataset         |     |        |       |         |           | N4-3        |         |       |         |
|      | Metric/         |     | Method |       | MLRVPST | PTDL-LSTM | PBLSTM      | PBDLSTM | PBKAN | PBxLSTM |
|      | RMSE            | tG  | ( ↓    | )     | 36.8    | 35.3      | 25.2        | 30.2    | 33.4  | 29.4    |
| RMSE |                 | tS  | fur    | ( ↓ ) | 29.4    | 25.5      | 8.7         | 23.8    | 27.4  | 21.9    |
| RMSE | tS              |     | obs    | ( ↓ ) | 61.2    | 53.2      | 67.5        | 62.8    | 51.8  | 42.1    |
|      | MAE             | tG  | ( ↓    | )     | 34.9    | 29.0      | 15.4        | 24.7    | 31.3  | 27.1    |
| MAE  | tS              | fur | (      | ↓ )   | 28.9    | 21.8      | 7.3         | 19.5    | 26.5  | 21.1    |
| MAE  | tS              | obs | (      | ↓ )   | 57.2    | 33.2      | 32.1        | 38.9    | 47.9  | 36.1    |
|      | mMAPE           | fr  | (      | ↓ )   | 30.7    | 40.3      | 24.6        | 28.2    | 60.0  | 30.6    |

performance, we employed teacher forcing during evaluation (providing ground truth values from previous time steps as inputs). This explains the improved metrics compared to auto-regressive real-world like inference from earlier tables.

We observed good convergence of PBMLP (Fig [4\)](#page-13-11), with the default setting mentioned in Appendix [A.4.](#page-18-0) Table [15](#page-24-2) shows performance with different hidden layer configurations, with [50, 100, 200] providing competitive results. Here, [100] denotes one hidden layer with 100 neurons, [50, 100] denotes two hidden layers with 50, and 100 neurons respectively, and so on. The maximum values for each row (corresponding to a metric) are shown in bold. In Table [16,](#page-24-2) we vary the batch size in our method. We found a batch size of 64 to provide an optimal performance for our experiments. In our exploration of activation functions, ReLU, SiLU, and Mish exhibited similar performance, with ReLU proving more robust across batch sizes (Table [18\)](#page-24-3).

We also examined all possible combinations of the regularizer weights λebv and λebs. Table [17](#page-24-3) highlights extreme cases where one regularizer is set to zero while the other is at a higher value, i.e., keeping only the EBV term by setting λebv = 0.1 and λebs = 0, and only the EBS term by setting λebv = 0 and λebs = 0.1. We found that performance is better while using both regularizers together rather than in isolation.

However, we found that excessively high values for the regularizers can compete with the regression loss terms, a common issue noted in PINN literature. Specifically, when λebs is set too high, it can significantly degrade performance due to the larger number of surface zones typically present in a furnace overpowering the loss function. Based on these observations and to avoid unnecessary complexity with varying values (e.g., 0.1, 0.3, etc), which resulted in minimal performance differences, we opted for a single value of λebv and λebs for the sensitivity analysis for both regularizers. This decision simplifies our design while ensuring optimal learning rate adjustments are considered. The results are presented in Figure [5](#page-14-0) where we observe a stable performance across values except a drop in R-MSE tG at λebs = 10 as mentioned.

### A.8 DATA DETAILS: FROM FURNACE TO ML MODEL TRAINING AND EVALUATION

We now discuss the data set details of our benchmarking. Prior to discussing the data used for ML model training and evaluation, we provide the reader a brief flavor on the physical understanding of a real-world furnace, along with its operation.

**1297 1299 1300 1301 1302 1303 1304 1305 1306 1307 1308 1309 1310 1311 1312 1313 1314 1315 1317 1319 1321 1324** 2 ### TRAINING ### 3 c r i t e r i o n = nn . MSELoss ( ) 4 o p t i m i z e r = optim . Adam ( model . p a r a m e t e r s ( ) , l r =LEARNING RATE) 5 f o r e i n tqdm ( r a n g e ( 1 , EPOCHS+1) ) : 6 model . t r a i n ( ) 7 f o r ( b a t c h i d x , s a m p l e b a t c h e d ) i n enumerate ( t r a i n l o a d e r E B V S ) : 8 # s a m p l e b a t c h e d [ 0 ] : data , s a m p l e b a t c h e d [ 1 ] : l a b e l s , s a m p l e b a t c h e d [ 2 ] : a u x v a r s 9 X t r a i n b a t c h = s a m p l e b a t c h e d [ 0 ] . t o ( d e v i c e ) 10 y t r a i n b a t c h = s a m p l e b a t c h e d [ 1 ] . t o ( d e v i c e ) 11 a u x v a r s d i c t b a t c h = s a m p l e b a t c h e d [ 2 ] 13 d f a G G t e n s o r b a t c h = a u x v a r s d i c t b a t c h [ ' d f a G G t e n s o r ' ] . t o ( d e v i c e ) 14 s g a r r p l u s h g t e n s o r b a t c h = a u x v a r s d i c t b a t c h [ ' s g a r r p l u s h g ' ] . t o ( d e v i c e ) 15 d f a S S t e n s o r b a t c h = a u x v a r s d i c t b a t c h [ ' d f a S S t e n s o r ' ] . t o ( d e v i c e ) 16 g s a r r p l u s h s t e n s o r b a t c h = a u x v a r s d i c t b a t c h [ ' g s a r r p l u s h s ' ] . t o ( d e v i c e ) 18 o p t i m i z e r . z e r o g r a d ( ) 20 y t r a i n p r e d = model ( X t r a i n b a t c h ) 21 t r l o s s r e g t m p s = c r i t e r i o n ( y t r a i n p r e d , y t r a i n b a t c h ) 23 ## EBV t e r m s 24 p b e b v p r e d = g e t p b e b v p r e d ( 25 s g a r r p l u s h g t e n s o r b a t c h , d f a G G t e n s o r b a t c h , 26 y t r a i n p r e d [ : , : n g a s z o n e s ] 27 ) 28 p b e b v a c t u a l = t o r c h . z e r o s ( p b e b v p r e d . s i z e ( ) ) . t o ( d e v i c e ) 30 ## EBS t e r m s 31 p b e b s p r e d = g e t p b e b s p r e d ( 32 g s a r r p l u s h s t e n s o r b a t c h , d f a S S t e n s o r b a t c h , 33 y t r a i n p r e d [ : , n g a s z o n e s : n g a s z o n e s + n f u r s u r f z o n e s + n o b s s u r f z o n e s ] 34 ) 35 p b e b s a c t u a l = t o r c h . z e r o s ( p b e b s p r e d . s i z e ( ) ) . t o ( d e v i c e ) 36 37 t r l o s s e b v = c r i t e r i o n ( p b e b v p r e d , p b e b v a c t u a l ) / y t r a i n p r e d . s i z e ( 0 ) 38 t r l o s s e b s = c r i t e r i o n ( p b e b s p r e d , p b e b s a c t u a l ) / y t r a i n p r e d . s i z e ( 0 ) <sup>40</sup> b a t c h l o s s = t r l o s s r e g t m p s + lambda ebv \* t r l o s s e b v + lambda ebs \* t r l o s s e b s 41 b a t c h l o s s . backward ( ) 42 o p t i m i z e r . s t e p ( )

**1334**

Algorithm 2 PyTorch-styled pseudo-code for training loop of our framework

Table 15: Performance of PBMLP (ReLU) variant of our

| method against    | varying | hidden   | layer    | configurations |       |                         |       |       |       |
|-------------------|---------|----------|----------|----------------|-------|-------------------------|-------|-------|-------|
| Hidden layer      |         |          |          |                |       |                         |       |       |       |
|                   | [100]   | [50,100] | [50,100, |                |       |                         |       |       |       |
| RMSE tG ( ↓ )     | 11.64   | 17.25    | 10.04    | 10.84          | 14.27 |                         |       |       |       |
| RMSE tS fur ( ↓ ) | 10.05   | 15.23    | 7.95     | 7.83           | 12.46 |                         |       |       |       |
| RMSE tS obs ( ↓ ) | 34.82   | 37.62    | 31.64    | 33.57          | 36.42 |                         |       |       |       |
| mMAPE fr ( ↓ )    | 8.76    | 9.15     | 6.84     | 8.06           | 7.51  |                         |       |       |       |
|                   |         |          |          |                |       | variant using different | batch | sizes |       |
|                   |         |          |          |                |       | RMSE tG ( ↓ )           | 12.70 | 10.04 | 10.73 |
|                   |         |          |          |                |       | RMSE tS fur ( ↓ )       | 9.14  | 7.95  | 9.69  |
|                   |         |          |          |                |       | RMSE tS obs ( ↓ )       | 39.75 | 31.64 | 31.79 |
|                   |         |          |          |                |       | mMAPE fr ( ↓ )          | 5.24  | 6.84  | 8.29  |

Table 16: Performance of the proposed PBMLP variant using different batch sizes .

Table 17: Effect of individual regularizer

| terms in PBMLP    |  |          |          |       |                   |       |       |       |       |       |
|-------------------|--|----------|----------|-------|-------------------|-------|-------|-------|-------|-------|
| Metric            |  | EBV only | EBS only | PBMLP |                   |       |       |       |       |       |
| RMSE tG ( ↓ )     |  | 11.85    | 11.66    | 10.04 |                   |       |       |       |       |       |
| RMSE tS fur ( ↓ ) |  | 10.36    | 11.07    | 7.95  |                   |       |       |       |       |       |
| RMSE tS obs ( ↓ ) |  | 32.46    | 32.04    | 31.64 |                   |       |       |       |       |       |
| mMAPE fr ( ↓ )    |  | 6.42     | 7.53     | 6.84  |                   |       |       |       |       |       |
|                   |  |          |          |       | Metric            | PBMLP |       |       |       |       |
|                   |  |          |          |       | RMSE tG ( ↓ )     | 10.04 | 13.57 | 10.07 | 15.26 | 10.16 |
|                   |  |          |          |       | RMSE tS fur ( ↓ ) | 7.95  | 8.86  | 8.02  | 14.02 | 7.71  |
|                   |  |          |          |       | RMSE tS obs ( ↓ ) | 31.64 | 39.65 | 31.64 | 36.23 | 31.63 |
|                   |  |          |          |       | mMAPE fr ( ↓ )    | 6.84  | 5.88  | 6.23  | 7.03  | 6.33  |

Table 18: Performance of PBMLP using different activation functions in the underlying network.

#### A.8.1 BACKGROUND ON FURNACE OPERATION

For experimentation, we consider a real-world, walking beam top-fired furnace in Swerim (former Swerea MEFOS), Sweden, which has been studied by Hu et al. [Hu et al.](#page-11-2) [\(2019\)](#page-11-2). Figure [6](#page-25-2) illustrates the furnace, which can be conceptually subdivided into several zones along both its length and height, such as dark, control, and soaking, which represent regions with distinct temperatures. It has varying

 ### HELPER FUNCTIONS ### # For EBV d f a G G t e n s o r a l l = g e t d f a A B t e n s o r a l l ( tea GG , g e t t o r c h f l o a t ( X t G g a s z o n e p r e v ) . t o ( d e v i c e ) ) s g a r r p l u s h g a l l = g e t s g a r r p l u s h g a l l ( g e t t o r c h f l o a t ( X hg ) . t o ( d e v i c e ) , tea GS , t o r c h . h s t a c k ( ( g e t t o r c h f l o a t ( X t S f u r n a c e p r e v ) , g e t t o r c h f l o a t ( X t S o b s t a c l e p r e v ) ) ) . t o ( d e v i c e ) ) d e f g e t p b e b v p r e d i n s t a n c e ( s g a r r p l u s h g t e n s o r , d f a G G t e n s o r , t G s i n g l e p r e d ) : ## computes \mathbf{v} g v e c t o r f o r one ti me s t e p d e f g e t p b e b v p r e d ( s g a r r p l u s h g t e n s o r b a t c h , d f a G G t e n s o r b a t c h , y t r a i n p r e d o n l y t G ) : ## c a l l s g e t p b e b v p r e d i n s t a n c e f o r a l l i n s t a n c e s i n t h e b a t c h # For EBS d f a S S t e n s o r a l l = g e t d f a A B t e n s o r a l l ( tea SS , g e t t o r c h f l o a t ( np . h s t a c k ( [ X t S f u r n a c e p r e v , X t S o b s t a c l e p r e v ] ) ) . t o ( d e v i c e ) ) g s a r r p l u s h s a l l = g e t g s a r r p l u s h s a l l ( g e t t o r c h f l o a t ( X hs ) . t o ( d e v i c e ) , tea SG , g e t t o r c h f l o a t ( X t G g a s z o n e p r e v ) . t o ( d e v i c e ) ) d e f g e t p b e b s p r e d i n s t a n c e ( g s a r r p l u s h s t e n s o r , d f a S S t e n s o r , t S s i n g l e p r e d ) : ## computes \mathbf{v} s v e c t o r f o r one ti me s t e p d e f g e t p b e b s p r e d ( g s a r r p l u s h s t e n s o r b a t c h , d f a S S t e n s o r b a t c h , y t r a i n p r e d o n l y t S ) : ## c a l l s g e t p b e b s p r e d i n s t a n c e f o r a l l i n s t a n c e s i n t h e b a t c h

![](_page_25_Figure_2.jpeg)

Algorithm 3 PyTorch-styled pseudo-code for helper functions in our framework

Figure 6: Illustration of the real-world furnace in Swerim, Sweden, and its subdivision as different zones [Hu](#page-11-2) [et al.](#page-11-2) [\(2019\)](#page-11-2). Figure is best viewed in color. The temperature increases towards the discharge end (at the right), as indicated by a darker shade. The slabs are heated while moving from the left to the right.

heights for different zones but is of fixed length and width. It has a target heating temperature of 1250 ◦C and its production capacity is 3 tonne/hr. Reheating furnaces are used to heat intermediate steel products usually known as stock (e.g., blooms, billets, slabs).

 ### HELPER FUNCTIONS ( s e t 2) ### d e f i n v e r s e t r a n s f o r m V e c t o r i z e d p t ( s c a l e d t e n s o r , range , min along dims , d i s t ) : range min , range max = r a n g e o r i g t e n s o r = m i n a l o n g d i m s + d i s t \*( s c a l e d t e n s o r − r a n g e m i n ) / ( range max − r a n g e m i n ) r e t u r n o r i g t e n s o r d e f g e t a n m a t t e n s o r ( t B s i n g l e r o w t e n s o r ) : t M a t t e n s o r = t o r c h . t i l e ( t B s i n g l e r o w t e n s o r , ( Ng , 1) ) c o e f b m a t T = c o e f b m a t . T f o r i i i n r a n g e ( c o e f b m a t T . shape [ 1 ] ) : # T a y l o r s e r i e s l oop bn= c o e f b m a t T [ : , [ i i ] ] b n t e n s o r = t o r c h . from numpy ( bn ) . f l o a t ( ) . t o ( d e v i c e ) i f i i ==0: a n m a t t e n s o r = t o r c h . mul ( t o r c h . t i l e ( b n t e n s o r , ( 1 , t M a t t e n s o r . s i z e ( 1 ) ) ) , t M a t t e n s o r \*\* i i ) e l s e : a n m a t t e n s o r += t o r c h . mul ( t o r c h . t i l e ( b n t e n s o r , ( 1 , t M a t t e n s o r . s i z e ( 1 ) ) ) , t M a t t e n s o r \*\* i i ) r e t u r n a n m a t t e n s o r d e f g e t p b e b v p r e d i n s t a n c e ( s g a r r p l u s h g t e n s o r , d f a G G t e n s o r , t G s i n g l e p r e d ) : s t a r t i d c o l , e n d i d c o l =0 , n g a s z o n e s t G c u r r e n t t e n s o r = i n v e r s e t r a n s f o r m V e c t o r i z e d p t ( t G s i n g l e p r e d , ( 0 , 1 ) , y t r m i n a l o n g d i m s [ [ 0 ] , s t a r t i d c o l : e n d i d c o l ] . t o ( d e v i c e ) , y t r d i s t [ [ 0 ] , s t a r t i d c o l : e n d i d c o l ] . t o ( d e v i c e ) ) g g a r r t e n s o r = t o r c h . sum ( t o r c h . mul ( d f a G G t e n s o r , s b c o n s \* t o r c h . t i l e ( t G c u r r e n t t e n s o r \*\*4 , ( d f a G G t e n s o r . s i z e ( 0 ) , 1) ) ) , 1 , keepdim=True ) . T a n m a t G t e n s o r = g e t a n m a t t e n s o r ( t G c u r r e n t t e n s o r ) tmpmat2= s b c o n s \* t o r c h . mul ( t o r c h . t i l e ( V i c u r r e n t t e n s o r , ( a n m a t G t e n s o r . s i z e ( 0 ) , 1 ) ) , t o r c h . t i l e ( t G c u r r e n t t e n s o r \*\*4 , ( a n m a t G t e n s o r . s i z e ( 0 ) , 1) ) ) tmpmat1= t o r c h . mul ( a n m a t G t e n s o r , t o r c h . t i l e ( c o e f k m a t T t e n s o r , ( 1 , a n m a t G t e n s o r . s i z e ( 1 ) ) ) ) g l e a v e t e n s o r = t o r c h . sum ( t o r c h . mul ( tmpmat1 , tmpmat2 ) , 0 , keepdim=True ) p b e b v p r e d i n s t a n c e = t o r c h . abs ( g g a r r t e n s o r + s g a r r p l u s h g t e n s o r −4\* g l e a v e t e n s o r ) p b e b v p r e d i n s t a n c e /= p b e b v p r e d i n s t a n c e . max ( dim =1 , keepdim=True ) [ 0 ] r e t u r n p b e b v p r e d i n s t a n c e

 Through a series of discrete pushes, the transport of slabs occurs within a furnace. As shown in Figure [6,](#page-25-2) a first slab at an ambient temperature is pushed from the charging end at the left side of furnace (lower temperature, shown in a lighter shade). At each push, all slabs move forward towards the discharge end at the right (higher temperature, shown in a darker shade). For a few specific regions in the furnace, the process operator pre-defines a few set point temperatures, which indicate the temperatures to which the slabs must be heated. The slabs once heated to the required set point temperatures, are collected at the discharge end. The movement of the slabs is controlled by the walk-interval (walk rate), depending on the desired throughput.

 The internal combustion is controlled via firing rates of a few burners located in specific regions. In Figure [6,](#page-25-2) we can see that there are six burners: 2 in each of control zones 1, 2, and 3. In this particular furnace, the pair of burners in a control zone share the same firing rate values. Note that these firing rates are normalized in [0, 1].

 Describing the behavior of a furnace state involves combustion models, control loops, set point calculations, and fuel flux control in zones. It also involves linearization and model order reduction for state estimation and state-space control. The inherent complexity makes the modeling a nonlinear dynamic system. We provide set point temperatures, walk interval, firing rates and initial state of the furnace (indicated by temperatures of various gas and surface regions/zones in it) as inputs to this system. These inputs, along with the overall movement of the slabs within the furnace, influence the mass and energy flow throughout the furnace system. This, in turn, results in a new furnace state, characterized by a new set of temperatures.

Algorithm 4 PyTorch-styled pseudo-code for additional helper functions in our framework

![](_page_27_Diagram_0.jpeg)

Figure 7: Illustration of flow of the data generation algorithm. The figure is best viewed in color. Dashed lines denote feedback from past time step. Blue/red/gray lines correspond for tG/tS/fr, respectively. Block Abbreviations are, FR: Firing Rate, FP: Flow-pattern, ENTH: Enthalpy, TRAN: Heat-transfer, COND: Conduction analysis, EBV/S: Energy-Balance Volume/Surface, and DFA: Directed Flux Area. Details of components present in the text.

The ideal scenario involves a computational model that can predict the next set of temperatures based on the provided inputs. This predicted state can then be compared to the desired set point temperatures. Deviations from the set points trigger adjustments in the firing rates. If a region's predicted temperature falls short of the set point, the firing rate for the corresponding burner increases. Conversely, if the predicted temperature exceeds the desired value, the firing rate is lowered. A Proportional-Integral-Derivative (PID) controller is employed to manage these adjustments in practice. This controller factors in the walk interval to ensure smooth and controlled changes in the firing rates, ultimately leading to a furnace state that aligns with the set point temperatures.

#### A.8.2 PROPOSED DATA GENERATION METHODOLOGY FOR TEMPERATURE PREDICTION USING ML

As shown in Figure [6,](#page-25-2) it is possible to conceptually divide the furnace into 1, 2, and 12 sections across its width, height, and length respectively. This results in a total of 24 volume/gas zones, where gaseous material could reside. These zones can be visualized using the dashed vertical and horizontal lines in the figure.

Additionally, at a time step, there can be 17 slabs inside the furnace, each of which has 6 surfaces, thus, resulting in 102 slab surfaces. With prior knowledge of the 3D structure of our furnace, we computed a total of 76 furnace walls, which could be called furnace surfaces. We can respectively call the 102 slab surfaces as obstacle/ slab surface zones, and the 76 furnace walls as furnace surface zones. Collectively, the obstacle/ slab surface zones and furnace surface zones result in a total of 178 surface zones, which in addition to the volume zones form the basis of utilization of the Hottel's zone method.

The flow of combustion products within the furnace results in heat release. This causes radiation interchange among all possible pairs of zones: gas to gas, surface to surface, and surface to gas (and vice-versa). The dominating heat transfer mechanism in such processes is Radiative Heat Transfer (RHT), which naturally occurs among the other heat transfer mechanisms: conduction and convection. For each pair of zones, there would be an energy balance, i.e., the amount of energy entering a zone would equal the amount leaving it. To model the RHT, the zone method subdivides an enclosure into a finite number of isothermal volume and surface zones, and applies energy balance to each of them. In our case, for example, we have a total of 202 zones (178 for surfaces and 24 for volumes).

We can model the radiative exchange among any two zones by leveraging underlying governing physical equations, and *energy balances*. The zone method also employs pre-computed exchange areas (which are general forms of view factors). The main objective is to then compute unknown parameters such as temperatures (of volumes and surfaces), and heat fluxes. This could be done by solving a set of simultaneous equations. We direct the interested reader to [Yuen & Takara](#page-13-2) [\(1997\)](#page-13-2); [Hu et al.](#page-10-1) [\(2016;](#page-10-1) [2019\)](#page-11-2), for a better perspective of the zone method.

**1517**

**1519**

**1521**

**1534**

**1554**

We shall design the data framework in such a way that it can easily plug in any standard ML (or DL) model for regression. For this, notice that although the various entities within the computational method depend on the geometry of the furnace, we can make a learnable model agnostic of the geometry, if we can train it by simply using data in the form of input-output pairs, and (optional) auxiliary/ intermediate variables (say, for regularization).

One simple way is to collect all relevant values from across zones corresponding to an entity in the form of a vector. For example, we could collect all gas zone temperatures within a vector, and likewise, for other entities such as surface zones, enthalpies, heat fluxes, node temperatures, etc, we could form individual vectors. This gives us the freedom to ignore the 3D structure during training as we can simply deal with vectors and their mappings, say within a neural network, or any other ML technique. Post-inference analysis or fine-grained process control could later be performed via our knowledge of which zone an attribute of the vector maps to.

In Figure [7,](#page-27-2) we present our proposed algorithmic flow mimicking the Hottel's zone method [Hottel &](#page-10-12) [Cohen](#page-10-12) [\(1958\)](#page-10-12); [Hottel & Saforim](#page-10-13) [\(1967\)](#page-10-13); [Yuen & Takara](#page-13-2) [\(1997\)](#page-13-2) based computational model of Hu et al. [Hu et al.](#page-10-1) [\(2016\)](#page-10-1), for data generation aimed at training regression-based ML models. In this, notice how we represent all the relevant entities as vectors. While we shall discuss all relevant terms of the zone method in detail, during the explanation of the modeling part, we now briefly give an overview of the various stages of the zone method. Here, let Φ represents a particular block/ stage, and θ represents the applicable parameters for the underlying function (abbreviated name shown in the subscript). Following are the stages in the generation method (represented by a block in Figure [7\)](#page-27-2):

- 1. Firing Rates updation block (Φθfr ): Using the predicted gas (tG) and surface (tS) zone temperatures from a previous time step, a calibration against the setpoint temperatures provided in sp is performed to update the firing rates fr for the current time step (also denoted as f). In Figure [7](#page-27-2) we use slightly abused notations of fr and sp to represent firing rates and setpoints for avoiding confusion with other notations such as *surface*.
- 2. DFA block (Φθdfa ): Notice that for a time step, the inputs tS, t<sup>G</sup> are obtained from the corresponding values obtained as outputs in the previous time step, shown respectively by dashed red and blue backward arrows. Here, |S| and |G| denote the total number of surface and gas zones, and, t<sup>S</sup> ∈ <sup>R</sup> |S| , t<sup>G</sup> ∈ <sup>R</sup> |G| are vectors collecting all the surface zone and gas/volume zone temperatures respectively. Hu et al [Hu et al.](#page-10-1) [\(2016\)](#page-10-1), using an updated Monte-Carlo based Ray-Tracing (MCRT) algorithm [Matthew et al.](#page-12-11) [\(2014\)](#page-12-11), provide fixed, pre-computed Total Exchange Areas (TEAs) (forms of view factors [Yuen & Takara](#page-13-2) [\(1997\)](#page-13-2)) as inputs along with tS, tG, for computing the Radiation Exchange factors, or the Directed Flux Area (DFA) terms. The TEAs are denoted as: GS ∈ R |G|×|S|×N<sup>g</sup> , SS ∈ R |S|×|S|×N<sup>g</sup> , GG ∈ R |G|×|G|×N<sup>g</sup> , and SG ∈ R |S|×|G|×N<sup>g</sup> (we can drop the third dimension for the sake of brevity). Here, GS, SS, GG, and SG contain the pre-computed gas-surface, surface-surface, gas-gas, and surface-gas exchange areas. ↼ GS ∈ R |G|×|S| , ↼ SS ∈ R |S|×|S| , ↼ GG ∈ R |G|×|G| , and ↼ SG ∈ |S|×|G| are the corresponding DFA terms for GS, SS, GG, and SG respectively (↼ indicates the direction of flow). Here, N<sup>g</sup> denotes the number of gases used for representing a real gas medium. Initially, we assume that a steady-state has been reached, and hence assign ambient temperature values to tS, tG. The parameters θdfa represent fixed correlation coefficients (as discussed in the methodology section).
- 3. Flow pattern (Φθfp ) and enthalpy blocks (Φθenth ): Given initial firing rates in f ∈ <sup>R</sup> |B| (|B| is a function of the number of burners), the block representing the function Φθfp obtains the flow pattern flat(F), which is further used by the block representing the function Φθenth to obtain the enthalpy vector q. Note that, the flow of combustion gases within an enclosure causes mass flow into (+ve) and out (-ve) of a zone, for each inter-zone boundary plane. This flow could be pre-computed in a CPU instantly using a polynomial fitted through isothermal CFD simulations that define a range of experimental points, derived with Box–Behnken designs [Ferreira et al.](#page-10-16) [\(2007\)](#page-10-16). The flow pattern resulted is by nature a matrix F ∈ R <sup>|</sup>G|×<sup>12</sup>, but the spatial dependency among the matrix elements can be discarded for simplicity, and we can rather represent an equivalent flattened vector flat(F) ∈ <sup>R</sup> <sup>12</sup>|G<sup>|</sup> obtained in row-major fashion. Note that, as already mentioned, we subdivide an enclosure into several cubes/ boxes (zones in our

| 1566 |                                                                                   |
|------|-----------------------------------------------------------------------------------|
|      | Algorithm 5 Data generation algorithm for a fixed furnace configuration           |
| 1:   | Initialize a steady-state furnace configuration via set points and walk interval. |
| 2:   | Initialize X = {} , T > 0 (max no. of steps). (0) (0) (0)                         |
| 3:   | Initialize t                                                                      |
|      | G , t                                                                             |
|      | S with steady-state ambient temperatures, and f                                   |
| 4:   | for t=1 to T do ▷ t: time-step                                                    |
| 5:   | f                                                                                 |
|      | ( t ) ← Φ θfr ( f                                                                 |
|      | ( t − 1)                                                                          |
|      | , set point temperatures , t                                                      |
|      | ( t − 1)                                                                          |
|      | G , t                                                                             |
|      | ( t − 1) ) S                                                                      |
| 6:   | q                                                                                 |
|      | ( t ) ← Φ θenth (Φ θfp ( f                                                        |
|      | ( t ) ))                                                                          |
| 7:   | ↼ ↼ ↼ ↼                                                                           |
|      | GG ( t )                                                                          |
|      | GS ( t )                                                                          |
|      | SG ( t )                                                                          |
|      | SS ( t ) ← Φ θdfa ( t                                                             |
|      | ( t − 1)                                                                          |
|      | G , t                                                                             |
|      | ( t − 1)                                                                          |
|      | , GG , GS , SG , SS ) , , , S ↼ ↼                                                 |
| 8:   | t                                                                                 |
|      | ( t )                                                                             |
|      | G ← Φ θebv ( q                                                                    |
|      | ( t )                                                                             |
|      | GG ( t )                                                                          |
|      | GS ( t ) ) , , ↼ ↼                                                                |
| 9:   | w ( t ) ← Φ θtran ( t                                                             |
|      | ( t )                                                                             |
|      | G , t                                                                             |
|      | ( t − 1)                                                                          |
|      | SS ( t )                                                                          |
|      | SG ( t ) ) , , S                                                                  |
| 10:  | t                                                                                 |
|      | ( t )                                                                             |
|      | S ← Φ ebs ( n                                                                     |
|      | ( t )                                                                             |
|      | ) , where n                                                                       |
|      | ( t ) ← Φ θcon ( w ( t ) )                                                        |
| 11:  | X t ← { f                                                                         |
|      | ( t )                                                                             |
|      | , F                                                                               |
|      | ( t )                                                                             |
|      | , q                                                                               |
|      | ( t )                                                                             |
|      | , t                                                                               |
|      | ( t )                                                                             |
|      | , t                                                                               |
|      | ( t )                                                                             |
|      | G , w ( t )                                                                       |
|      | , n                                                                               |
|      | ( t ) } S                                                                         |
| 12:  | X ← X ∪ X t                                                                       |
| 13:  | end for                                                                           |
| 14:  | return X                                                                          |

Algorithm 5 Data generation algorithm for a fixed furnace configuration

case). Since any cube has 6 surfaces, and for each surface we have two directions of flow (+ve and -ve), this results in 12 flows for each volume zone, and thus, the 12 arises in the dimensionality of F.

Also, for each volume zone i, we would require an enthalpy transport term (Q˙ enth)<sup>i</sup> . We introduce an enthalpy vector q ∈ R |G| to compactly represent these terms.

- 4. Energy Balance Volume (EBV) block (Φθebv ): We introduce a block to compute the volume zone temperatures t<sup>G</sup> using the enthalpy vector q and the DFA terms ↼ GG and ↼ GS.
- 5. Heat transfer block (Φθtran ): Together with the volume zone temperatures tG, the obtained DFAs ( SS, ↼ SG), and the previously obtained (or initialized) surface zone temperatures tS, we obtain the heat transfer/ flux to the surfaces as a variable w.
- 6. Conduction analysis block (Φθcon ): The heat flux on each surface zone serves as a boundary condition for performing a conduction analysis, to compute the transient heat conduction through each surface. The conduction process results in the node temperatures, which we represent as a variable n.
- 7. Energy Balance Surface (EBS) block (Φθebs ): The computation of heat transfer/ flux and surface zone temperatures are coupled together as the surface energy balance equations. Having computed the heat transfer and performing the conduction analysis, the surface zone temperatures in t<sup>S</sup> can be updated using the node temperatures n. This is a fixed function.

The Algorithm: Algorithm [5](#page-29-0) presents the steps involved in the data generation method. We assume that for a steady-state furnace configuration (with fixed set points and walk interval), our data set is in the form: X = {Xt} T <sup>t</sup>=1, where, X<sup>t</sup> = {f (t) ,F (t) , q (t) , t (t) S , t (t) <sup>G</sup> , <sup>w</sup>(t) , n (t)} is the set of observed variables as described in Figure [7,](#page-27-2) for a time-step t. Note that the computations of flow patterns, enthalpy, and node temperatures can be treated independently from the energy balance equations.

| timestep | firing_rates | walk_interval       | setouts | flowattern             | g_enthalpy                                             | ts_gaseno                                                 | ts_furnace                                               | ts_obstacle                                               | w_flux_furnace                                     |  |
|----------|--------------|---------------------|---------|------------------------|--------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------|--|
| 0        | 1000035      | [0,162, 0.9, 0.689] | 750     | [905,0,1220.0, 1250.0] | [0.27214, 0.00037, 0.0, 0.0, 0.15124, 0.00502,...]     | [325971.875, 6805.781, 16632.312, 20740.859, 783.621,...] | [1238.396, 655.898, 669.693, 720.935, 729.375,...]       | [899.66, 696.459, 276.871, 707.375, 267.441, 244.599,...] | [1227.219, 61.728, 44.997, 77.785, 123.674, 26...] |  |
|          |              |                     |         |                        |                                                        | [0.27379, 6803.078, 16803.453, 20947.594, 785.105,...]    | [201.843, 696.454, 676.84, 707.373, 277.841,...]         | [1470.822, 138.764, 84.222, 121.113, 176.747, ...]        |                                                    |  |
|          |              |                     |         |                        |                                                        | [0.27392, 6803.078, 16803.453, 20947.594, 785.105,...]    | [201.843, 696.454, 676.84, 707.373, 277.841,...]         | [1470.822, 138.764, 84.222, 121.113, 176.747, ...]        |                                                    |  |
|          |              |                     |         |                        |                                                        | [0.27392, 6803.078, 16803.453, 20947.594, 785.105,...]    | [201.843, 696.454, 676.84, 707.373, 277.841,...]         | [1470.822, 138.764, 84.222, 121.113, 176.747, ...]        |                                                    |  |
| 1        | 1000050      | [0,176, 0.9, 0.697] | 750     | [905,0,1220.0, 1250.0] | [0.27392, 6803.078, 16803.453, 20947.594, 785.105,...] | [0.27532, 6849.953, 676.223, 723.702, 727.41,...]         | [301.243, 696.504, 676.845, 707.41, 288.102, 262.75,...] | [1460.180, 121.778, 121.823, 162.165, 226.299,...]        |                                                    |  |
|          |              |                     |         |                        |                                                        | [0.27532, 6849.953, 676.223, 723.702, 727.41,...]         | [301.243, 696.504, 676.845, 707.41, 288.102, 262.75,...] | [1460.180, 121.778, 121.823, 162.165, 226.299,...]        |                                                    |  |
|          |              |                     |         |                        |                                                        | [0.27532, 6849.953, 676.223, 723.702, 727.41,...]         | [301.243, 696.504, 676.845, 707.41, 288.102, 262.75,...] | [1460.180, 121.778, 121.823, 162.165, 226.299,...]        |                                                    |  |
|          |              |                     |         |                        |                                                        | [0.27532, 6849.953, 676.223, 723.702, 727.41,...]         | [301.243, 696.504, 676.845, 707.41, 288.102, 262.75,...] | [1460.180, 121.778, 121.823, 162.165, 226.299,...]        |                                                    |  |

Figure 8: Sample training data instances for each time step within a configuration.

Figure [8](#page-29-1) illustrates a few sample time steps (in rows), and the corresponding entities (in columns) generated by using Algorithm [5.](#page-29-0) The full list of entities that we generate for a time step is: 'timestep', 'firing rates', 'walk interval', 'setpoints', 'flowpattern', 'q enthalpy', 'tG gaszone', 'tS furnace', 'tS obstacle', 'w flux furnace',

**1624**

**1627**

**1629**

**1657**

| to_gasonome_prev | ts_furnace_prev                                    | ts_obstacle_prev                                  | firing_rates                                      | to_gasonome         | ts_furnace                                                 | ts_obstacle                                        | firing_rates_next                                 |                     |
|------------------|----------------------------------------------------|---------------------------------------------------|---------------------------------------------------|---------------------|------------------------------------------------------------|----------------------------------------------------|---------------------------------------------------|---------------------|
| 0                | [1230.741, 654.484, 668.378, 719.49, 782.103, ...  | [898.918, 696.524, 676.938, 707.417, 759.248, ... | [272.753, 190.658, 221.352, 256.900, 235.417, ... | [0.162, 0.9, 0.689] | [1238.396, 655.898, 669.693, 720.935, 783.621, ...         | [899.669, 696.459, 676.871, 707.375, 759.241, 8... | [282.33, 198.022, 230.603, 267.441, 244.599, 2... | [0.176, 0.9, 0.689] |
| 1                | [1238.396, 655.898, 669.693, 720.935, 783.621, ... | [899.66, 696.459, 676.871, 707.375, 759.241, 8... | [282.33, 198.022, 230.603, 267.441, 244.599, 2... | [0.176, 0.9, 0.697] | [1245.547, 657.297, 670.983, 722.349, 785.105, ...         | [900.576, 676.84, 707.373, 759.285, 8...           | [291.843, 203.389, 239.773, 277.841, 253.712, ... | [0.188, 0.9, 0.705] |
| 2                | [1245.547, 657.297, 670.983, 722.349, 785.105, ... | [900.576, 696.454, 676.84, 707.373, 759.285, 8... | [291.843, 205.389, 239.773, 277.841, 253.712, ... | [0.188, 0.9, 0.705] | [1252.052, 658.657, 672.223, 723.702, 786.523, ...         | [901.643, 696.504, 676.845, 707.41, 759.375, 8...  | [301.287, 214.661, 288.102, 262.75, 2...          | [0.197, 0.9, 0.721] |
| 3                | [1252.052, 658.657, 672.223, 723.702, 786.523, ... | [901.643, 696.504, 676.845, 707.41, 759.375, 8... | [301.287, 212.751, 248.861, 288.102, 262.75, 2... | [0.197, 0.9, 0.712] | [1257.793, 656.953, 673.385, 724.94, 787.842, ...          | [902.832, 696.606, 676.883, 727.482, 759.508, ...  | [310.652, 220.1, 257.862, 298.222, 271.709, 27... | [0.209, 0.9, 0.781] |
| 4                | [1257.793, 659.953, 673.385, 722.964, 787.842, ... | [902.832, 696.606, 676.883, 782.092, 759.508, ... | [310.652, 220.1, 257.862, 298.222, 271.709, 27... | [0.209, 0.9, 0.718] | [1286.848, 661.255, 676.956, 707.59, 726.984, 789.244, ... | [904.15, 696.761, 676.954, 707.59, 759.686, 82...  | [311.959, 217.441, 266.784, 208.212, 280.599, ... | [0.218, 0.9, 0.727] |

'w flux obstacle', 'nodetmp 1d furnace', 'nodetmp 2d obstacle'. The names of the entities are self-explanatory (e.g., 'nodetmp 1d furnace' refers to 1D node temperatures for furnace surfaces, 'nodetmp 2d obstacle' refers to 2D node temperatures for obstacle surfaces), where G as usual, denotes *gas zone* and S denotes *surface zone*, the latter, is further divided into *furnace* and *obstacle*.

Figure 9: Rearranged training data instances (selected columns).

Assuming that the original data is stored in a Pandas DataFrame (using a Python syntax), for each time step we also need the following entities: 'firing rates next', 'tG gaszone prev', 'tS furnace prev', and 'tS obstacle prev'. This is because, for computing the entities in a time step, we make use of the temperatures in the previous time step. At the same time, for experimental purposes, we also try to directly predict the next firing rate via ML. Thus, using Python syntax, we could perform the following:

a) df['firing rates next'] = df['firing rates'].shift(-1) followed by df = df.drop(df.tail(1).index). b) df['tG gaszone prev']=df['tG gaszone'].shift(1), df['tS furnace prev'] = df['tS furnace'].shift(1), df['tS obstacle prev'] = df['tS obstacle'].shift(1) followed by df = df.drop(df.head(1).index).

The rearranged data can be visualized as in Figure [9](#page-30-1) (we only showcase relevant entities here, owing to limited space). Essentially, we add a new column 'firing rates next' by shifting the original firing rates column a step back and then dropping the last row. Likewise, we add new columns for *prev* temperatures by shifting the original temperature columns a step forward and then dropping the first row. Please note that some additional auxiliary variables are used by the computational method of Hu et al. [Hu et al.](#page-10-1) [\(2016\)](#page-10-1), which are mostly constants, and could thus be repeated/ copied for each time step. They are: 'corrcoeff b', 'Qconvi', 'extinctioncoeff k', 'gasvolumes Vi', 'QfuelQa sum', 'surfareas Ai', 'emissivity epsi', 'convection flux qconvi'. We later leverage them in training our PCNN, with the help of regularizers.

Now we can form any data set containing N samples: X = {(x (i) , y (i) )} N <sup>i</sup>=1 to train an off-the-shelf, standard ML/ DL model fθ(.) with learnable parameters θ, which expects an input instance x (i) as vector and predicts an output vector y (i) , i.e., y (i) = fθ(x (i) ). Here, x (i) and y (i) can be formed using entities from desired columns obtained from the rearranged data as shown in Figure [9.](#page-30-1) Notice how the above proposed ML training framework via our data generation in the form of simple inputoutput pairs lets any generic regression model learn freely without requiring 3D geometry-specific knowledge during the training. This makes our proposed framework geometry-agnostic, and hence flexible by nature to accommodate any ML method.

# A.8.3 BENCHMARKING DATA SET DETAILS FOR ML MODEL DEVELOPMENT AND EVALUATION

Algorithm [5](#page-29-0) outlines data generation for a fixed furnace configuration (defined by set points and walk interval). Set points are desired temperatures for certain zones. We represent a configuration as: SP1 SP2 SP3 WI, where SP1, SP2, SP3 and WI respectively denote the set point 1, set point 2, set point 3, and walk interval. Under normal conditions naturally occurring in practice, following will hold true: SP1<SP2<SP3. For robustness, we consider 50 configurations (based on the furnace in Fig [6\)](#page-25-2) and generate corresponding *configuration datasets*, including abnormal configurations with arbitrary set points. Since each dataset has a unique configuration, their inherent data distributions differ.

From the 50 distinct datasets, we combine configurations (e.g., first, fourth, seventh) to form a consolidated training split. Similar combinations create validation and test splits with no overlap between them. This creates a test bed to evaluate model generalization across different data distributions, crucial for real-world deployment where inference data might differ from training data. Table [19](#page-31-1) details these configurations, indicating their membership in training, validation, or test splits, within parentheses. Test datasets (e.g., N1-2, N1-3) are named based on their set point characteristics and are also shown in bold.

It should be noted that the default SP1,SP2,SP3,WI setting is kept: 955 1220 1250 750. With this, we vary each of SP1, SP2, SP3, and WI with certain step-size. This leads to four groups/types of configurations within the Normal Behaviour Configurations shown in Table [19.](#page-31-1) The nomenclature of the test data sets is done to indicate their grouping, e.g., prefixes N1-, N2-, N3- and N4- denote whether the configuration belongs to the group with varying SP1, SP2, SP3, and WI respectively. Thus, Ni-j indicates the j-th configuration of the group i, and is used to represent a test *configuration data set*. As it can be seen, there are 11 normal test data sets where we evaluate the ML models.

Table 19: Benchmark data details.

| Normal Behaviour Configuration (SP1<SP2<SP3) |                                 |                                 |                                 |  |  |
|----------------------------------------------|---------------------------------|---------------------------------|---------------------------------|--|--|
| Type 1 (Varying SP1 only)                    | Type 2 (Varying SP2 only)       | Type 3 (Varying SP3 only)       | Type 4 (Varying WI only)        |  |  |
| 995.1220.1250.750 (Training)                 |                                 |                                 |                                 |  |  |
| 915.1220.1250.750 (Val)                      | 955.1170.1250.750 (Training)    | 955.1220.1230.750 (Training)    | 955.1220.1250.675 (Training)    |  |  |
| <b>925.1220.1250.750 (N1-1)</b>              | 955.1180.1250.750 (Val)         | 955.1220.1240.750 (Val)         | <b>955.1220.1250.705 (N4-1)</b> |  |  |
| 935.1220.1250.750 (Training)                 | <b>955.1190.1250.750 (N2-1)</b> | <b>955.1220.1250.750 (N3-1)</b> | 955.1220.1250.720 (Training)    |  |  |
| 945.1220.1250.750 (Val)                      | 955.1200.1250.750 (Training)    | 955.1220.1260.750 (Training)    | 955.1220.1250.735 (Val)         |  |  |
| <b>965.1220.1250.750 (N2-2)</b>              | 955.1210.1250.750 (Val)         | 955.1220.1270.750 (Val)         | <b>955.1220.1250.765 (N4-2)</b> |  |  |
| 975.1220.1250.750 (Training)                 | <b>955.1230.1250.750 (N2-2)</b> | <b>955.1220.1280.750 (N3-2)</b> | 955.1220.1250.780 (Training)    |  |  |
| 985.1220.1250.750 (Val)                      | 955.1240.1250.750 (Training)    | 955.1220.1290.750 (Training)    | 955.1220.1250.750 (Val)         |  |  |
| <b>995.1220.1250.750 (N1-3)</b>              |                                 | <b>955.1220.1300.750 (N3-3)</b> | <b>955.1220.1250.810 (N4-3)</b> |  |  |
|                                              |                                 |                                 | 955.1220.1250.825 (Training)    |  |  |

Table 20: Benchmark data details (abnormal configurations).

| Type | 1    |      |         | Abnormal Behaviour Arbitrary SPs Type 2 Type 3 Type 4 Type 5 |
|------|------|------|---------|--------------------------------------------------------------|
| 955  | 1220 | 1200 | 750.csv | (Training)                                                   |
| 955  | 1220 | 1210 | 750.csv | (Val)                                                        |
|      | 955  | 1220 | 1220    | 750.csv                                                      |
| 955  | 1250 | 1220 | 750.csv | (Training)                                                   |
| 955  | 1250 | 1220 | 765.csv | (Val)                                                        |
|      | 955  | 1250 | 1250    | 750.csv                                                      |
| 955  | 1260 | 1250 | 750.csv | (Training)                                                   |
|      | 955  | 1270 | 1250    | 750.csv                                                      |
|      |      |      |         | 1220 1250 955 750.csv (Training)                             |
|      |      |      |         | 1220 1250 955 795.csv                                        |
|      |      |      |         | 1220 955 1250 750.csv (Training)                             |
|      |      |      |         | 1220 955 1250 780.csv                                        |
|      |      |      |         | 1250 955 1220 750.csv (Training)                             |
|      |      |      |         | 1250 955 1220 825.csv                                        |
|      |      |      |         | 1250 1220 955 750.csv (Training)                             |
|      |      |      |         | 1250 1220 955 810.csv                                        |

Table [20](#page-31-2) details the remaining 16 configurations representing abnormal conditions (arbitrary set points). These are split for training and validation to make the model robust during training (similar to adversarial learning). We set aside 7 configurations apart from training/validation. A well-trained physics-aware model should perform poorly on these, rendering them unnecessary for testing.

For training a DL model, we aggregate the configuration datasets belonging to training splits as shown in Table [19.](#page-31-1) Prior to collecting, each of the datasets are reformatted to obtain time-shifted input-output pairs as discussed in the data generation methodology. After that rows of these training datasets are shuffled and stacked together to train the model. Each configuration is stored by a .csv file containing 1500 time steps sampled with a 15s delay, to account for conduction analysis. Thus, each configuration accounts for 6.25h worth data. Considering all 50 datasets, our generated data sets consists of 312.5h (or roughly, 13 days) of furnace data. We observed diminishing returns on model

**1731**

**1734**

**1737**

**1751**

**1754**

**1764**

**1767**

performance with further data size increases, justifying our decision to focus on this efficient data volume.

During time-shifted input-output pairs formation from a configuration dataset, we drop the first and last rows resulting in 1498 rows, to account for the shift operations. Thus, by consolidating the 20 training datasets, we get a total of 29960 train rows. These can be packed within a standard DataLoader in a framework like PyTorch, and train an off-the-shelf DL model. We can similarly obtain 17976 val rows, and also 26964 test rows (from across normal and abnormal configurations, if desired). We have reported results on the 11 datasets individually, where a model trained is used for auto-regressive, sequential prediction of subsequent time steps.

The discussed data sets, along with necessary data pre-processing, model training/evaluation scripts are provided in the following github repository <https://github.com/>, which shall be updated periodically to reflect the latest changes as available (while adhering to FAIR guidelines [\(Wilkinson](#page-12-15) [et al.,](#page-12-15) [2016\)](#page-12-15)). As a highlight, we provide the *configuration datasets* as separate .csv files. We also provide the consolidated stacked data as a .npz file. Furthermore, we also provide the TEA data as individual files, which are used during model training.

### A.9 POTENTIAL REAL-LIFE APPLICATIONS OF THE WORK AND ITS IMPACT

We now discuss how our method for furnace temperature profiling can be applied in various industries and contribute to energy efficiency and reduced emissions.

Steel and Metal Manufacturing: Our model can be directly applied to improve the efficiency of reheating furnaces used in steel and metal manufacturing processes. By providing accurate real-time temperature predictions, operators can optimize fuel consumption and reduce energy waste, leading to significant cost savings and lower carbon footprint. The ability to precisely control temperature profiles can also enhance product quality and consistency.

Glass and Ceramic Production: In the glass and ceramic industries, furnaces are crucial for melting, annealing, and tempering processes. Our model can be adapted to these furnace types, enabling tighter temperature control, reduced energy usage, and minimized defects. This can translate to higher productivity, lower operational costs, and a greener manufacturing process.

Cement and Lime Production: High-temperature furnaces are essential in cement and lime manufacturing for calcination and clinker production. Our physics-aware deep learning approach can be leveraged to optimize these processes, reducing fuel consumption and emissions while maintaining product quality. This can contribute to the sustainability efforts of cement and lime producers.

Petrochemical Refining: Furnaces are widely used in petrochemical refineries for various processes such as crude oil distillation, catalytic cracking, and reforming. By implementing our model, refineries can enhance energy efficiency, minimize fuel wastage, and lower greenhouse gas emissions. This can help refineries meet stringent environmental regulations while maintaining profitability.

# A.10 LIMITATIONS AND FUTURE WORK

Incorporation of Geometry-Specific Regularization: Future research should investigate the integration of geometry-specific regularization terms into our model. This could involve developing customized regularization strategies that account for the unique thermal characteristics of various furnace designs. By tailoring the model to specific configurations, we can potentially enhance its predictive accuracy and applicability across different industrial scenarios. This is beyond the scope of our work, which could be treated as a starting point in this direction.

Exploration of Foundational Models: Our approach could serve as a foundation for developing models that can be adapted for other related use cases. We envision leveraging techniques such as few-shot learning, continual learning, or transfer learning to enable our model to learn from limited data in new contexts. This would allow for rapid adaptation to different operational conditions and requirements, making our model more versatile and applicable across various industries.

Engineering aspects of Integration with Real-Time Monitoring Systems: Extensive study of challenges involved during engineering integration in a monitoring system could itself be another future direction of study, especially for a varied set of industries and furnace configurations.