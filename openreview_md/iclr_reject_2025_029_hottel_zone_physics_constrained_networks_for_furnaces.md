000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Anonymous authors Paper under double-blind review

## Abstract

This paper investigates a novel approach to improve the temperature profile prediction of furnaces in foundation industries, crucial for sustainable manufacturing.

While existing methods like the Hottel Zone model are accurate, they lack real-time inference capabilities. Deep learning methods excel in speed and prediction but require careful generalization for real-world applications. We propose a regularization technique that leverages the Hottel Zone method to make deep neural networks physics-aware, improving prediction accuracy for furnace temperature profiles. Our approach demonstrates effectiveness on various neural network architectures, including Multi-Layer Perceptrons (MLP), Long Short-Term Memory (LSTM), Extended LSTM (xLSTM) and Kolmogorov-Arnold Networks (KANs). We also discussion the data generation involved.

## 1 Introduction

Majority of economically relevant industries (automobiles, machinery, construction, household appliances, chemicals, etc) are dependent on the Foundation Industries (FIs) that provide crucial and foundational materials like glass, metals, cement, ceramics, bulk chemicals, paper, steel, etc. FIs are heavy revenue and employment drivers, for instance, FIs in the United Kingdom (UK) economy are worth £52B (EPSRC report), employ 0.25 million people, and comprise over 7000 businesses (IOM3 report). However, despite their economic significance, the FIs leverage energy-intensive methods within their furnaces. This makes FIs major industrial polluters and the largest consumers of natural resources across the globe. For example, in the UK, they produce 28 million tonnes of materials per year, and generate 10% of the entire UK's CO2 emissions (EPSRC report; IOM3 report). Similarly, in China, the steel industry accounted for 15% of the total energy consumption, and 15.4% of the total CO2 emissions (Zhang et al., 2018; Liang et al., 2020). These numbers put a challenge for the FIs in meeting our commitment to reduce net Green-House Gas (GHG) emissions, globally. With a closer look at any process industry (e.g., steel industry), one can observe that at the core, lies the process of conversion of materials (e.g., iron) into final products. This is done using a series of unit processes (Yu et al., 2007) involving steps such as dressing, sintering, smelting, casting, rolling, etc (see Qin et al. (2022) for an illustration). The equipment in such process industries operates in high-intensity environments (e.g., high temperature), and has bottleneck components such as reheating furnaces, which require complex restart processes post-failure. This causes additional labor costs and energy consumption. Thus, for sustainable manufacturing, it is important to monitor the temperature profile, and thus, the operating status of the furnaces. (Hu et al., 2019) have shown promise in achieving notable fuel consumption reduction by reducing the overall heating time.

Yuen & Takara (1997) in their study, have proved the elegance and superiority of the Hottel Zone method over counterparts to model the physical phenomenon of Radiative Heat Transfer (RHT) in high-temperature processes. Hu et al. (2016) proposed a computational model workflow based on the Hottel Zone method, and showed superiority over surrogate computational alternatives in terms of predictive performance. However, none of these approaches are suitable for real-time inference in modeling a furnace temperature profile. Deep Learning (DL) based neural network methods excel in achieving superior predictive performance and speed. Nonetheless, their generalization capabilities require special attention, particularly in critical real-world applications. In our work, we propose to revisit the Hottel Zone method and devise a novel regularization technique that could be used as a plug-and-play module to make a neural network physics-constrained (or

# Hottel Zone Physics-Constrained Networks For Furnaces

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 physics-aware) with regard to the underlying phenomena of high-temperature processes in furnaces. We show that for a time-step in a furnace, given a certain set of input entities, we could predict the desired output temperature entities more accurately (in terms of regression metrics) using our regularization technique, as opposed to using a vanilla neural network. We demonstrate the prowess of our proposal on different types of neural network architectures: Multi-Layer Perceptron (MLP) or feed-forward networks, sequential models such as Long Short-Term Memory (LSTM) based Recurrent Neural Networks (RNNs), as well as recently proposed Kolmogorov-Arnold Networks (KANs) and Extended LSTM (xLSTM). This work makes two key contributions: **Tensor-based Reformulation and Physics-Aware Neural** Networks: We reformulate the Hottel Zone Method's Directed Flux Areas (DFAs) and Energy Balance (EB) equations in tensor format, enabling neural network training. We further introduce a novel regularization technique that imbues the network with physics-awareness. **Extensive Experimental** Validation: We comprehensively validate the proposed approach using various neural network architectures. To this end, we suggest a dataset and benchmarking protocol (details provided in Section A.8). A github repository is maintained at https://github.com/ to facilitate real-time updates to the same as and when made.

Numerous real-world applications, including chemical reactors (Feng & Han, 2012), solar energy (Muhich et al., 2016; Marti et al., 2015), and 3D printing (Tran & Lo, 2018; Zhou et al., 2009), involve high-temperature processes exceeding 700◦C. These processes rely heavily on Radiative Heat Transfer (RHT) as a dominant mechanism alongside conduction and convection. Notably, RHT remains crucial for thermal transport even in vacuum conditions encountered in astronomical applications. We envision that our learnings could perhaps be extended to those applications with bespoke approaches. Due to space constraints, we have limited the length of the introduction section. Please refer to Section A.1 for a more detailed discussion, particularly regarding the motivation behind our research.

## 2 Related Work

In Section A.2, we provide a detailed discussion of related works. Due to space limitations, we will focus here on how our approach significantly differs from existing methods.

1. **View factor methods**: Existing methods Ebrahimi et al. (2013); Melot et al. (2011); Hu et al. (2018); Li (2005) simplify the modeling area and are geometry-specific. We propose a generic, geometry-agnostic model encompassing all exchange areas (radiation transfer interfaces).

2. **Neural network methods**: Existing methods Yuen (2009); Tausendschon & Radl ¨ (2021);
Garc´ıa-Esteban et al. (2021); Zhai & Zhou (2020); Zhai et al. (2023); Halme Stahlberg ˚ (2021); de Souza Lima et al. (2023); Liao et al. (2009); Hwang et al. (2019); Chen et al. (2022); Bao et al. (2023) often use simple MLPs, which lack generalization due to limited physics understanding. We introduce a Physics-constrained Neural Network (**PCNN**) framework that outperforms MLP and can be applied to other architectures like LSTM, KAN, xLSTM.

3. **Furnace temperature profiling**: Existing methods Kim & Huh (2000); Kim (2007); Jang et al. (2010); Tang et al. (2017); Nguyen et al. (2014); Hu et al. (2017); Ban et al. (2023); Li et al. (2023); Zanoli et al. (2023); Yu et al. (2022) focus on specific regions, while our method targets complete furnace temperature profiling, including gas zones, furnace walls, and slab surfaces. Our utilized data is more holistic. Existing neural methods in this category also lack physics awareness.

4. **PINNs**: Compared to the existing body of Physics-Informed Neural Network (PINN)
literature Raissi et al. (2019); Karniadakis et al. (2021); Drgona et al. ˇ (2021); Shen et al. (2023); Cai et al. (2021); Kim et al. (2022); Zhao et al. (2020); He et al. (2021); Boca de Giuli (2023); Han et al. (2023); Bunning et al. ¨ (2022); Park (2022); Wang et al. (2023);
Lahariya et al. (2022); Jing et al. (2023), we propose a novel variant specifically designed for zone method based modeling in reheating furnaces. Our approach is the first to utilize physics-constrained regularizers based on the zone method for temperature prediction. It requires minimal data (input-output pairs) and makes no geometry assumptions. Our data creation method is holistic and unique, encompassing all exchange areas. Our method, as we will see later, is based on a set of simultaneous equations to incorporate physics-awareness, and directly does not involve a differential equation. Thus, we call it a physics-constrained method, though PINN could be also used philosophically.

## 3 Proposed Method 3.1 Background

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 The Hottel Zone method subdivides a furnace into zones (volumes and surfaces) to predict Radiative Heat Transfer (RHT). Volume and Gas (G) zone is used interchangeably. Surface (S) zones are of two types, SF: furnace and SO: obstacle (e.g., slabs that are heated). Each zone has a uniform temperature. Sets of Energy-Balance (EB) equations govern radiation exchange between zones, considering incoming and outgoing radiation fluxes. These equations are iteratively updated to obtain the entire furnace's temperature profile. Following are the **key concepts**:
1. **Total Exchange Areas (TEAs)**: Pre-computed values representing the total area for radiation exchange between zone pairs (SS: surface-surface, SG/GS: surface-gas, GG: gas-gas).

2. **Directed Flux Areas (DFAs)**: Derived from TEAs and used to calculate radiant exchange between zone pairs at each step of the zone method.

3. **Weighted Sum of Grey Gases (WSGG) model**: Handles non-grey gases by representing them as a mixture of grey gases and a clear gas.

To account for our formulation of a neural network based approach, we first introduce the following four tensors to collectively represent the above TEAs: GS ∈ R
|G|×|S|×|Ng|, SS ∈ R
|S|×|S|×|Ng|, GG ∈ R
|G|×|G|×|Ng|, SG ∈ R
|S|×|G|×|Ng|. Here, |G|, |S| respectively denote the number of gas/
volume zones, and number of surface zones. In practice, |Ng| gases representing real gas medium are used, and hence, a third dimension has also been used in the above tensors. As discussed above, TEAs are pre-computed constants, used as inputs to our model. Slightly abusing notations, we can refer to a TEA by considering only the first two dimensions (for a pair of zones).

## 3.2 Exchange Area Calculation

The first step in the Zone method involves computation of Exchange Factors (Yuen & Takara, 1997).

The exchange factor among a pair of volume zones Vi and Vj is expressed as:

$$g_{i}g_{j}=\int_{V_{i}}\int_{V_{j}}{\frac{k_{i}k_{j}e^{-\tau}d V_{i}d V_{j}}{\pi r^{2}}}$$

$$(1)$$
πr2(1)
Physically, it represents the energy radiated from Vi and absorbed/ scattered by Vj . Here, k denotes the respective extinction coefficient, τ is the optical thickness among differential volume elements dVi and dVj , and r =p(xi − xj )
2 + (yi − yj )
2 + (zi − zj )
2. Now, let ni and nj respectively be unit normal vectors of dAi and dAj (corresponding to two surface zones Ai and Aj ). Then, the exchange factors gisj (between volume zone Vi and surface zone Aj ) and sisj (between surface zone Ai and surface zone Aj ), can be expressed as:

$$g_{i}s_{j}=\int_{V_{i}}\int_{A_{j}}\frac{k_{i}|\mathbf{n}_{j}.r|e^{-\tau}dV_{i}dA_{j}}{\pi r^{3}};s_{i}s_{j}=\int_{A_{i}}\int_{A_{j}}\frac{|\mathbf{n}_{i}.r||\mathbf{n}_{j}.r|e^{-\tau}dA_{i}dA_{j}}{\pi r^{4}}\tag{2}$$

Numerical evaluation of the above equations being complex, has led to analytical approximations, by considering an enclosure as a cube-square system, i.e, by representing a volume as a cube, and a surface as a square. This facilitates the tabulation of a "generic" set of exchange factors, which are applicable for most practical industrial geometries, using an updated Monte-Carlo based Ray-Tracing (MCRT) algorithm (Matthew et al., 2014). To this end, such pre-computed generic values are refered to as Total Exchange Areas (TEA), and we denote them by: GiSj , SiSj , GiGj and SiGj . Here, SiGj = GiSj . Note that throughout the text, G(or g) and S(or s) shall indicate terms corresponding to Gas/Volume, and Surface respectively.

## 3.3 Introducing Tensor Notations For Hottel Zone Method Based Neural Network

a11 … a1s … a1|S| t1
 … t s 
 … t|S| broadcast( a1 T a ) 1 T
a11 … a1s … a1|S| tS
T
{ b. , 1}
.

.

.

.

.

.

a11 … a1s … a1|S| |G|
{ b. , n }
an1 … a ns 
… an|S| .

. .

. . .

a11 … a1s … a1|S|
. . .

{ b. , Ng

}
aNg1 … aNgs … aNg|S| aNg T 
|S| aNg1 … aNgs … aNg|S|
.

 .

 .

. . .

☉ |G| |G| aNg1 … aNgs … aNg|S| Element-wise product
☉
aNg1 … aNgs … aNg|S| broadcast( aNg T )
|Ng | .

 .

 .

. . .

Channel-wise sum |S| TEA(GS)
|G|

$$(3)$$
$$\quad(4)$$

DFA(GS)
162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Using (3), (4, (5), and with GS as a reference, we make use of Figure 1 to illustrate the derivation of a compact matrix form for computing a DFA term efficiently for getting training samples of a neural network. Let, (GS)n be the n th slice of GS along the third dimension, and an = ˜bn(tS).

broadcast(a
⊤
n) reshapes a
⊤
nto the same dimension as (GS)n, i.e., R
|G|×|S|. tS ∈ R
|S|is a vector containing all the surface zone temperatures (in a time step), such that its j th entry tS(j) = Ts,j .

The j th entry an(j) of an ∈ R
|S|is computed using the function ˜bn with the correlation coefficients
{bi+1,n}
Ng i=0 as the parameters, and by following eq (5). We can also assume similar vector containing all gas zone temperatures (in a time step) tG ∈ R
|G|, with j th entry tG(j) = Tg,j .

4 WSGG is a method used to represent the absorptivity/ emissivity of real combustion products with a mixture of a couple of grey gases plus a clear gas, i.e, the number of grey gases is equal to Ng − 1. For each gas indexed by n, we have a set of pre-computed correlation coefficients {bi+1,n}
Ng i=0 for both gas and surface related coefficients, and an absorption coefficient kg,n. Then, the weighting coefficient ag,n(Tg,j ) (for gas-zone temperatures) and the weighting coefficient as,n(Ts,j ) (for surface-zone temperatures) can be expressed as a Nth g order polynomial in Tg,j (or Ts,j ):

$$a_{g,n}(T_{g,j})=\sum_{i=0}^{N_{g}}b_{i+1,n}T_{g,j}^{i};a_{s,n}(T_{s,j})=\sum_{i=0}^{N_{g}}b_{i+1,n}T_{s,j}^{i}\tag{5}$$

Here, ↼ indicates the direction of flow. Tg,j and Ts,j denote the temperatures for the j th volume and surface zones respectively, and are the values we want our model to predict (at each time step). Note that the collective representation of the DFAs can be expressed as:↼
GS ∈ R
|G|×|S|,↼
SS ∈ R
|S|×|S|,
↼
GG ∈ R
|G|×|G|,↼
SG ∈ R
|S|×|G|. In Eq (3)-(4), the TEA terms correspond to a particular grey gas being used, for example, (GiGj )k=knrepresents the TEA GiGj with the n th gas.

SS3):  ${\hat{G_iG_j}=\sum_{n=1}^{N_g}a_{g,n}(T_{g,j})(\overline{G_iG_j})_{k=k_n};\hat{S_iS_j}=\sum_{n=1}^{N_g}a_{s,n}(T_{s,j})(\overline{S_iS_j})_{k=k_n}}$  ${\hat{G_iS_j}=\sum_{n=1}^{N_g}a_{s,n}(T_{s,j})(\overline{G_iS_j})_{k=k_n};\hat{S_iG_j}=\sum_{n=1}^{N_g}a_{g,n}(T_{g,j})(\overline{S_iG_j})_{k=k_n}}$  Note that the location of ${\mathbf{S_i}}$ are ${T_{s,n}}$ and ${T_{s,n}}$ denote the temperature of the ${i^{th}}$ and ${n}$. 
Then, the **DFA terms related to gas-zone temperatures** can be expressed as:

$$\tilde{\mathbf{G}}\mathbf{S}=\sum_{n=1}^{N_{g}}(\overline{G}\mathbf{S})_{n}\odot\text{broadcast}(\mathbf{a}_{n}^{\top});\,\tilde{\mathbf{G}}\mathbf{G}=\sum_{n=1}^{N_{g}}(\overline{G}\mathbf{G})_{n}\odot\text{broadcast}(\tilde{b}_{n}(\mathbf{t}_{G})^{\top}).$$  and, the **DFA terms related to surface-zone temperatures** can be expressed as:
⊤). (6)
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 With the above DFA terms at our disposal, we can compute the gas/volume and surface zone temperatures at each time step of furnace operation by respectively using Energy-Balance Volume (EBV) and Energy-Balance Surface (EBS) equations. EBV and EBS are a set of simulataneous equations to capture the governing physics of RHT Hu et al. (2016). Figure 2 visually illustrates computation of the terms g(g)arr, s(g)arr and g*leave* involved in the EBV equation to compute the gas zone temperatures of a time step.

Let, g(g)arr ∈ R
|G| be a vector whose i th entry represents the amount of radiation arriving at the i th gas zone from all the other gas zones, s(g)arr ∈ R
|G|, a vector whose i th entry represents the amount of radiation arriving at the i th gas zone from all the other surface zones, g*leave* ∈ R
|G|, a vector whose i th entry represents the amount of radiation leaving the i th gas zone, and hg ∈ R
|G|a heat term. Also, let Tg,j (or Tg) and Ts,j (or Ts) denote the j th gas and surface zone temperatures respectively. Then, following EBV equations, the i th entries of g(g)arr, s(g)arr, g*leave* and hg can be computed as:

$$\begin{array}{l l}{{\mathbf{g}_{(g)a r r}(i)=\sum_{j}^{|G|}\tilde{\mathbf{G}_{i}}\tilde{\mathbf{G}_{j}}\sigma T_{g,i}^{4};}}&{{\mathbf{s}_{(g)a r r}(i)=\sum_{j}^{|S|}\tilde{\mathbf{G}_{i}}\tilde{\mathbf{S}_{j}}\sigma T_{s,i}^{4}}}\\ {{\mathbf{g}_{t e a v e}(i)=\sum_{n}^{|N_{g}|}a_{g,n}(T_{g,i})k_{g,n}\sigma V_{i}T_{g,i}^{4}}}&{{\mathbf{h}_{g}(i)=-(\dot{Q}_{c o n v})_{i}+(\dot{Q}_{f u e l,n e t})_{i}+(\dot{Q}_{a})_{i}+\mathbf{q}_{i}(\dot{Q}_{a})_{i},}}\end{array}$$
$$(8)$$

Here, the constants (known apriori) (Q˙conv)i, (Q˙*fuel,net*)i, and (Q˙a)i respectively denote the convection heat transfer, heat release due to input fuel, and thermal input from air/ oxygen. An enthalpy vector q ∈ R
|G|is computed using the flow-pattern obtained via polynomial curve fitting during simulation. σ is the Stefan-Boltzmann constant, Viis volume of i th gas zone.

$$\tilde{\mathbf{S}}\mathbf{S}=\sum_{n=1}^{N_{\mathbf{a}}}(\overline{\mathbf{S}\mathbf{S}})_{n}\odot\text{broadcast}(\tilde{b}_{n}(\mathbf{t}_{S})^{\top});\,\tilde{\mathbf{S}}\mathbf{G}=\sum_{n=1}^{N_{\mathbf{a}}}(\overline{\mathbf{S}\mathbf{G}})_{n}\odot\text{broadcast}(\tilde{b}_{n}(\mathbf{t}_{G})^{\top}).\tag{7}$$

## 3.4 Energy-Balance Based Physics-Regularization

Let, s(s)arr ∈ R
|S|, be a vector whose i th entry represents the amount of radiation arriving at the i th surface zone from all the other surface zones, g(s)arr ∈ R
|S|, a vector whose i th entry represents

t1
 … tj … t|G| Energy-balance volume tG
T
. .

.

. .

. ∑ j |S| DFA(GiSj ) σ Ts,j 4 s(g)arr t1

 … t j 

 … t |G| tG
T
. . .

. . . 

∑ j |G| DFA(GiGj ) σ Tg,j 4 g(g)arr t1
 … tj … t|S| tS
T
|G| |S| t1 4

 … tj 4

 … t|G| 4 t1 4
 … tj 4
 … t |S| 4 V1 t1 4
 … V|G|t |G| 4 a11 kg1

 … 
 

 a1|G|kg1 t1 4
 … tj 4
 … t|G| 4
. . . 

|G| ☉ σ t1 4
 … tj 4
 … t |S| 4
. . . 

V1 t1 4

 … V|G|t |G| 4

.

. ☉ σ .

|G| ☉ σ aNg1 kgNg

 … aNg|G|kgNg

. . .

DFA(GG)
DFA(GS)
Row-sum Row-sum

$$(\mathbf{6})$$

Column-sum, Transpose
. . .

. . . 

∑ n Ng a g,n
(Tg,i)kg,n σ ViTg,i 4 gleave Figure 2: Derviation of the matrix forms of the EBV equations for physics based regularizers.
270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 the amount of radiation arriving at the i th surface zone from all the other gas zones, s*leave* ∈ R
|S|, a vector whose i th entry represents the amount of radiation leaving the i th surface zone, and hs ∈ R
|S| a heat term. Then, following EBS equations, the i th entries of s(s)arr, g(s)arr, s*leave* and hs can be computed as:

$$\begin{array}{l l}{{{\mathbf{s}_{(s)a r r}}(i)=\sum_{j}^{|S|}{\mathbf{\tilde{s_{i}}}{\mathbf{S_{j}}}\sigma T_{s,j}^{4}};}}&{{{\mathbf{g}_{(s)a r r}}(i)=\sum_{j}^{|G|}{\mathbf{\tilde{s_{i}}}{\mathbf{\tilde{G_{j}}}}\sigma T_{g,j}^{4}}}}\\ {{{\mathbf{s}_{l e a v e}}(i)=A_{i}\epsilon_{i}\sigma T_{s,i}^{4};}}&{{\qquad\quad\mathbf{h}_{s}(i)=A_{i}(\dot{q}_{c o n v})_{i}-\dot{Q}_{s,i}}}\end{array}$$

$$(9)$$
$$(10)$$

For a surface zone i, the constants (known apriori) Ai( ˙q*conv*)i and Q˙s,i respectively denote the heat flux to the surface by convection and heat transfer from it to the other surfaces. Here, Aiis the area, and ϵiis the emissivity of the i th surface zone.

The calculated terms in the Energy-Balance (EB) equations represent the heat entering and leaving each zone. In simpler terms, these equations ensure an energy balance by placing all incoming heat terms on the left-hand side (LHS) and outgoing terms on the right-hand side (RHS). Leveraging these terms in an optimization framework allows us to minimize the difference between LHS and RHS. To achieve this, we introduce the following terms:

 $\textbf{v}_g=(\textbf{g}_{(g)arr}+\textbf{s}_{(g)arr}-4\textbf{g}_{leave}+\textbf{h}_g)\in\mathbb{R}^{|G|}$  $\textbf{v}_s=(\textbf{s}_{(s)arr}+\textbf{g}_{(s)arr}-\textbf{s}_{leave}+\textbf{h}_s)\in\mathbb{R}^{|S|}$
Here, |G|/|S| denotes the number of Gas/ Surface zones. Intuitively, vg and vs are vector representatives corresponding to EBV and EBS. Let, λebv, λebs > 0 are hyper-parameters corresponding to Lebv and Lebs, such that Lebv=||normalize(vg)||22is our proposed regularizer term corresponding to the EBV. Similarly, Lebs=||normalize(vs)||22is our proposed regularizer term corresponding to the EBS. We use: normalize(v) = v/max(v), where max(v) is the maximum value from among all components in v. The core idea is to leverage the Energy Balance (EB) equations, which represent well-established physical laws governing heat transfer in the furnace. These equations enforce a balance between incoming and outgoing heat for each zone. The vectors vg and vs capture the residuals between the incoming and outgoing heat terms in the EB equations for gas (g) and surface (s) zones, respectively. By minimizing the L2 norm of these residuals (after normalization), we are essentially penalizing the network for deviating significantly from the physical constraints imposed by the EB equations. This encourages the network to learn temperature profiles that adhere to these well-defined energy balances. Minimizing the L2 norm encourages the network to drive all components of the residual vectors towards zero. The normalization step ensures all zones contribute equally to the penalty, regardless of their absolute temperature values. This prevents zones with naturally higher temperatures from dominating the regularization term.

## 3.5 Putting Together The Neural Network Objective

We now discuss the design of our final neural network. We formulate the objective in such a way that we can plug the above proposed regularizers in a standalone neural network architecture trained to regress output temperatures given a set of easily available input entities at each time step of a furnace operation. While starting the furnace operation, ambient temperatures are readily available (depicting the *initial state of the furnace*), along with walk interval, desired target set point temperatures. Then, based on the firing rates chosen for the burners of the furnace, there would be a resulting flow pattern in the furnace. This is a result of heat flow, and mass flow within the furnace (mass flow happens because of the slab movements, which need to be heated). This flow pattern would cause a change in the overall enthalpy, leading to a new temperature profile (*new state*) of the furnace, which can be measured by the resulting new gas and surface zone temperatures. These temperatures in turn could serve as input temperatures for the next step's prediction. For a more intuitive understanding of furnace operation, please refer Section A.8. In a practical setup, a neural network deployed could expect to consume the previous step temperatures, firing rates, walk interval, and set point temperatures as inputs. The output could then be the new 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

$$(11)$$
$$L_{\mathrm{sup}}$$

Here, Lsup = E(x(i),y(i))∈X [||y
(i) − fθ(x
(i))||22] is a standard *supervised term for regression*. To make such a network physics-aware, all we need to do is include the above proposed terms Lebv and Lebs into the final objective. It should be noted that, in doing so, we do not need to make any architectural changes to the network in terms of inputs and outputs. Also, all auxiliary variables used in computation of (8) and (9) are only used during training of a physics-aware network, and are not required in the inference. The regularization terms are computed using additional vectors as described earlier, influence the learning because they have the temperature terms in them. For example, in (10), vg depends on gas zone temperatures Tg,j via g(g)arr, g*leave* in (8). While computing Lebv we obtain the Tg,j terms using the network output, which are associated with the computational graph and thus help the updates during back-propagation. On the other hand, s(g)arr is associated with Ts,j which are detached for back-propagation while updating gas zone temperatures.

Similarly, in (10), vs depends on surface zone temperatures Ts,j via s(s)arr, s*leave* in (9). While computing Lebs we obtain the Ts,j terms using the network output, which are associated with the computational graph and thus help the updates during back-propagation. On the other hand, g(s)arr is associated with Tg,j which are detached for back-propagation while updating surface zone temperatures. The overall physics-aware loss is formulated as:
Ltotal = Lsup + λebvLebv + λebsLebs (12)

$$\mathcal{L}_{total}=\mathcal{L}_{sup}+\lambda_{ebv}\mathcal{L}_{ebv}+\lambda_{ebs}\mathcal{L}_{ebs}$$
$\eqref{eq:walpha}$. 
When calculating the physics-aware loss terms we detach certain temperature terms associated with one zone type (e.g., surface zone temperatures) during updates of the other zone type (e.g., gas zone temperatures). This prevents the network from altering these relationships unnaturally during backpropagation. As analogy, we can refer to a Teacher-Student Learning setup: Imagine the network learning from a teacher (the EB equations) that provides the correct temperature relationships. Detaching specific terms allows the network to focus on learning the mapping between furnace inputs and its own predicted zone temperatures, while still adhering to the guidance provided by the teacher (the EB equations) through the physics-aware loss terms. Algorithm 1 provides detailed steps of our proposed approach.

## Algorithm 1 Algorithm Of The Proposed Method

1: **Input:** X ={(x
(i), y
(i))}
N
i=1, furnace configuration (set points and walk interval). *maxeps >* 0.

2: Initialize θ, TEAs, λebv, λebs > 0.

3: Initialize tG ∈ R
|G|, tS ∈ R
|S| with ambient temperatures, and firing rates.

4: for EN=1 to *maxeps* do ▷ EN: Epoch No. 5: for i=1 to N do ▷ i: time step 6: Compute DFAs↼
GG(t),
↼
GS(t),
↼
SG(t),
↼
SS(t)using (6) and (7).

7: Compute Lebv using (8) and (10). 8: Compute Lebs using (9) and (10). 9: Compute Lsup using X .

10: θ
(i) ← θ
(i−1) − η∇θL*total* ▷ Using (12)
11: **end for** 12: **end for** 13: θ
∗ ← θ N.maxeps 14: **return** θ
∗

## 4 Experiments

In this section we report results on 11 datasets obtained using different configurations of a real-world furnace based on Hu et al. (2019) (details in Section A.8.3). Major objective of the experiments is temperatures, and the next firing rates as well. With input-output data X ={(x
(i), y
(i))}
N
i=1 acquired in this manner, we can estimate parameters θ of a neural network fθ(.) by training it to predict y
(i)
given x
(i), for all time step i, as:
θ
∗ ← arg min θ Lsup (11)
378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

| Table 1: Comparison of proposed methods on the N1-2 Dataset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |      |                   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|-------------------|
| Dataset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | N1-2 | 965 1220 1250 750 |
| Metric/ Method MLP PBMLP LSTM PBLSTM DLSTM PBDLSTM KAN PBKAN xLSTM PBxLSTM RMSE tG (↓) 113.4 35.6 33.0 26.7 117.1 32.4 24.3 22.6 130.6 29.3 RMSE tS fur (↓) 116.4 22.4 25.6 11.7 114.4 24.9 15.2 14.6 119.1 20.4 RMSE tS obs (↓) 106.9 43.4 61.1 66.5 109.3 67.4 35.1 33.6 139.8 45.4 MAE tG (↓) 89.5 28.2 27.4 16.9 100.9 27.2 21.4 19.9 129.1 26.8 MAE tS fur (↓) 96.2 17.8 21.5 9.9 101.1 20.1 14.3 13.8 118.6 19.5 MAE tS obs (↓) 79.9 29.6 39.4 31.4 86.9 44.4 29.8 29.3 136.3 39.8 mMAPE fr (↓) 176.6 58.5 29.5 23.5 201.0 26.2 44.2 32.6 200.8 27.8 |      |                   |

Table 2: Comparison of proposed methods on the N2-1 Dataset

Dataset N2-1 955 1190 1250 750

Metric/ Method MLP PBMLP LSTM PBLSTM DLSTM PBDLSTM KAN PBKAN xLSTM PBxLSTM

RMSE tG (↓) 121.1 45.4 36.8 37.0 123.4 28.3 29.5 **18.0** 95.5 33.0

RMSE tS fur (↓) 123.8 27.6 29.5 28.9 120.5 18.7 20.7 8.8 80.6 24.9

RMSE tS obs (↓) 113.1 52.4 65.6 63.3 114.5 51.9 41.0 **27.2** 90.7 51.7 

MAE tG (↓) 96.9 38.8 31.3 31.4 106.9 19.7 26.2 **15.4** 93.5 30.3 

MAE tS fur (↓) 103.6 24.8 26.7 25.5 106.4 16.5 19.8 7.7 80.1 24.1 MAE tS obs (↓) 87.4 39.9 46.2 44.2 92.2 **21.9** 35.9 22.9 86.6 46.5

mMAPE fr (↓) 187.6 67.8 28.4 29.8 210.6 **24.9** 43.7 34.2 212.3 26.2

to consider different neural network architectures with and without our proposed regularizers (and keeping everything else constant). Any gains reported could be attributed to our proposed regularizers that seek to enhance the physics-awareness of a network. Results across all the 11 datasets are reported in Tables 6, 7, 8, 9. For neural network architectures, we study following variants: MLP, LSTM, a stacked/deep LSTM (DLSTM) and recently proposed KAN and xLSTM. We use commonly used regression performance metrics such as RMSE and MAE for the temperature prediction. We also report MAPE additionally for predicting the next firing rates (MAPE is more suitable due to the range of values that firing rates take). A metric against each of the different entities has been reported. For example, RMSE tS fur denotes the average RMSE for all the furnace surface zone predictions, RMSE tS obs denotes the average RMSE for all the obstacle surface zone predictions, RMSE tG denotes the average RMSE for all the gas zone predictions. mMAPE fr indicates the performance on the firing rate predictions. For all metrics, a lower value indicates a better performance. All metrics are reported along the rows of a table, and the columns represent the different methods. For each row, the best performing metric corresponding to a method is shown in bold. In Table 1 we report the performance of the architectures MLP, LSTM, DLSTM, KAN and xLSTM on the N1-2 dataset. We also report performances of PBMLP, PBLSTM, PBDLSTM, PBKAN and PBxLSTM, which are the Physics-Based (PB) variants of MLP, LSTM, DLSTM, KAN and PBxLSTM respectively. The green colored cells indicate that a PB variant has obtained a better performance than a vanilla variant without our proposed regularizers. Compared to the simpler MLP, we could see massive gains by the PBMLP. The DLSTM (and xLSTM) variant possibly tends to overfit due to stacking of more LSTM layers, and performs worse compared to a vanilla LSTM model. Stacking LSTMs offered no advantage likely due to the data's inherent structure. Unlike language tasks that benefit from complex LSTM modeling with longer windows/time steps, zone-based method only requires capturing the relationship between the current state (s(i)) and the next (s(i+1)). Our data generation (details in Appendix) captures the relationship between current state (s(i)) and next state (s(i+1)), making complex LSTM architectures unnecessary. Initial experiments confirmed this, showing no significant improvement with longer windows compared to the simpler s(i), s(i+1) pairs. This aligns with Occam's razor - favoring simpler models with comparable performance. However, when equipped with our regularizers, the PBDLSTM (and PBxLSTM) method obtains much better performance than the DLSTM (and xLSTM). The vanilla LSTM which performs better than the MLP and DLSTM, also obtains improvements after using the physics based regularizers, as indicated by the performance of PBLSTM. We also notice KAN to perform better than the base MLP
(as observed in recent literature). In fact, the PBKAN variant performs the best among all methods at times. In Table 2 we report performances of the same approaches on the N2-1 dataset. We observed similar conclusions: the PB variants were outperforming their vanilla variants (as shown by green), thus depicting the benefit of the proposed regularizers. In this case, we observed that the PBKAN method obtains the best performance among all.

Table 3: Comparison of proposed methods on average across the datasets.

Dataset Average Metric/ Method MLP PBMLP LSTM PBLSTM DLSTM PBDLSTM KAN PBKAN xLSTM PBxLSTM
RMSE tG (↓) 79.2 35.1 37.2 30.4 83.5 27.9 26.1 **19.3** 85.2 31.7 RMSE tS fur (↓) 75.6 23.1 27.1 20.2 78.1 20.5 18.5 **12.4** 75.5 24.2 RMSE tS obs (↓) 86.8 49.5 64.9 64.1 89.9 61.7 37.0 **29.8** 95.3 45.8 MAE tG (↓) 62.2 29.1 29.7 23.8 70.9 22.4 23.8 **16.8** 83.4 29.5 MAE tS fur (↓) 62.6 20.3 23.1 18.1 68.9 17.3 18.0 **11.6** 74.9 23.5 MAE tS obs (↓) 62.5 33.9 40.7 38.6 65.3 36.0 33.4 **25.7** 90.9 40.5 mMAPE fr (↓) 119.3 53.6 39.2 26.7 141.8 **25.9** 46.4 39.3 131.4 37.5
432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Difference in the datasets N1-2 and N2-1 comes by varying setpoint temperatures of the first and second control zones of the furnace. This shows that depending on the furnace configuration of the same geometry, the performance of a deep learning model may vary as the data distribution changes due to the difference in underlying physical entities. However, if equipped with physics based regularizers, we could make the network adhere to the governing laws, and get a reasonable predictive performance. We further report on how the different methods perform across varying configurations or datasets on average, in Table 3. We observed similar performances, where the PB variants led to better performance. In Tables 6, 7, 8, 9 we report the performances of the compared approaches across all the 11 datasets. We noticed that not only the PB variants obtain a better performance throughout, they are also more stable across different datasets as indicated by their standard deviations. In Figure 4 we plot the convergence of our PBMLP method. Losses with respect to all the individual terms converge well. In Figure 3 we report visual plots of actual and predicted temperatures for PBMLP. We also show that omitting previous temperatures from the neural network inputs leads to an worse performance, thus, highlighting the impact of a furnace state on the model performance. We conducted a sensitivity analysis of λebv and λebs in Figure 5, observing stable performance across values.

## 4.1 Final Note On Impact Of Energy-Balance Regularization

Throughout the text, for all baseline methods in a column, the counterpart with the PB- prefix (eg, PBMLP, PBLSTM, PBDLSTM, PBKAN, PBxLSTM) indicates the usage of energy-balance regularization terms, and the green colored metrics all denote the consistent performance boost, as compared to the vanilla variants (eg, MLP, LSTM, DLSTM, KAN, xLSTM).

## 4.2 Comparison Against Recent State-Of-The-Art (Sota)

While we acknowledge the importance of contextualizing our work, we recognize that making direct comparisons is challenging due to the unique characteristics of our framework. Most existing methods in the literature focus on limited exchange areas in furnace temperature modeling. In contrast, our robust data generation framework encompasses the entire set of exchange areas, which is essential for accurate temperature profiling. To facilitate meaningful comparisons, we relate our results to established baselines recognized as State-Of-The-Art (SOTA) techniques in settings similar to ours. Specifically, we evaluate the impact of our research by comparing our proposed Physics-Based (PB) variants against the following methods: i) MLRVPST (Bao et al. (2023)) and ii) PTDL-LSTM (de Souza Lima et al. (2023)), the 9 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

| Table 4: Comparison of proposed methods on average across the datasets against recent SOTA. Dataset Average Metric/ Method MLRVPST (Bao et al. (2023)) PTDL-LSTM (de Souza Lima et al. (2023)) PBLSTM PBDLSTM PBKAN PBxLSTM RMSE tG (↓) 31.2 37.2 30.4 27.9 19.3 31.7 RMSE tS fur (↓) 24.5 27.1 20.2 20.5 12.4 24.2 RMSE tS obs (↓) 51.1 64.9 64.1 61.7 29.8 45.8 MAE tG (↓) 28.8 29.7 23.8 22.4 16.8 29.5 MAE tS fur (↓) 23.7 23.1 18.1 17.3 11.6 23.5 MAE tS obs (↓) 45.9 40.7 38.6 36.0 25.7 40.5 mMAPE fr (↓) 29.6 39.2 26.7 25.9 39.3 37.5   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

latter of which is comparable to our LSTM implementation. The results of the comparisons are presented in Table 4. We observed that our proposed variants outperform the SOTA in general. The full set of results are presented in Tables 11, 12, 13, and 14.

## 5 Conclusions

This work proposes a novel regularization technique that leverages the Hottel Zone method to make deep neural networks *physics-aware* for improved furnace temperature profile prediction. Our approach is effective across various network architectures, including Multi-Layer Perceptrons (MLPs), Long Short-Term Memory (LSTM) networks, Kolmogorov-Arnold Networks (KANs) and Extended LSTM (xLSTM), as evidenced on datasets based on real-world furnace configurations with varying set points. In Sections A.9 and A.10, we respectively discuss further real-life applications of our work, along with limitations of our work and future research directions.

## Acknowledgments

The authors wish to acknowledge ETHICS STATEMENT There are no ethical concerns related to our work. REPRODUCIBILITY STATEMENT Sections A.4, A.6, A.8.2, and A.8.3 respectively aim at ensuring reproducibility at the following four levels: 1. Architectural and training details (e.g. number of epochs, hyper-parameters used, etc), 2. PyTorch-styled code for understanding of the implementation, 3. Algorithmic methodology used to generate dataset for ML model training, and 4. Exact data set creations and splits used for training and evaluation, with details.

## References

Yunqi Ban, Xianpeng Wang, Guodong Zhao, and Jian Wu. *Multiobjective Operation Optimization of* Reheating Furnace based on Data Analytics, 2023. 2, 17 Qingfeng Bao, Sen Zhang, Jin Guo, Zhiqiang Li, and Zhenquan Zhang. Multivariate linear-regression variable parameter spatio-temporal zoning model for temperature prediction in steel rolling reheating furnace. *Journal of Process Control*, 123:108–122, 2023. 2, 9, 10, 18, 21 Felix Bunning, Benjamin Huber, Adrian Schalbetter, Ahmed Aboudonia, Mathias Hudoba de Badyn, ¨
Philipp Heer, Roy S Smith, and John Lygeros. Physics-informed linear regression is competitive with two machine learning methods in residential building mpc. *Applied Energy*, 310:118491, 2022. 2, 18 Shengze Cai, Zhicheng Wang, Sifan Wang, Paris Perdikaris, and George Em Karniadakis. Physicsinformed neural networks for heat transfer problems. *Journal of Heat Transfer*, 143(6):060801, 2021. 2, 18 Laura Boca de Giuli. *Physics-based neural network modelling, predictive control and lifelong* learning applied to district heating systems, 2023. 2, 18 Chien-Jung Chen, Fu-I Chou, and Jyh-Horng Chou. Temperature prediction for reheating furnace by gated recurrent unit approach. *IEEE Access*, 10:33362–33369, 2022. 2, 18 M De Beer, CG Du Toit, and PG Rousseau. A methodology to investigate the contribution of conduction and radiation heat transfer to the effective thermal conductivity of packed graphite pebble beds, including the wall effect. *Nuclear Engineering and Design*, 314:67–81, 2017. 16 Rodrigo de Souza Lima, Leonardo Azevedo Scardua, and Gustavo Maia de Almeida. Predicting ´
temperatures inside a steel slab reheating furnace using deep learning. *Seven Editora*, 2023. 2, 9, 10, 17, 21 540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Jan Drgo ´ na, Aaron R Tuor, Vikas Chandan, and Draguna L Vrabie. Physics-constrained deep learning ˇ
of multi-zone building thermal dynamics. *Energy and Buildings*, 243:110992, 2021. 2, 18 Hadi Ebrahimi, Akbar Zamaniyan, Jafar S Soltan Mohammadzadeh, and Ali Asghar Khalili. Zonal modeling of radiative heat transfer in industrial furnaces using simplified model for exchange area calculation. *Applied Mathematical Modelling*, 37(16-17):8004–8015, 2013. 2, 17 Heather N Emady, Kellie V Anderson, William G Borghard, Fernando J Muzzio, Benjamin J Glasser, and Alberto Cuitino. Prediction of conductive heating time scales of particles in a rotary drum. Chemical Engineering Science, 152:45–54, 2016. 16 EPSRC report. EPSRC report. https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?

GrantRef=EP/V026402/1, 2020. 1, 15 YT Feng and K Han. An accurate evaluation of geometric view factors for modelling radiative heat transfer in randomly packed beds of equally sized spheres. International journal of heat and mass transfer, 55:6374–6383, 2012. URL https://doi.org/10.1016/j. ijheatmasstransfer.2012.06.025. 2 SL Costa Ferreira, RE Bruns, Hadla Sousa Ferreira, Geraldo Domingues Matos, JM David, GC Brandao, EG Paranhos da Silva, LA Portugal, PS Dos Reis, AS Souza, et al. Box-behnken ˜ design: an alternative for the optimization of analytical methods. *Analytica chimica acta*, 597(2):
179–186, 2007. 29 Juan Jose Garc ´ ´ıa-Esteban, Jorge Bravo-Abad, and Juan Carlos Cuevas. Deep learning for the modeling and inverse design of radiative heat transfer. *Physical Review Applied*, 16(6):064006, 2021. 2, 17 Daniel Halme Stahlberg. Digital twin of a reheating furnace, 2021. ˚ 2, 17 Jiawei Han, Mehrdad Mesgarpour, Lazarus Godson Asirvatham, Somchai Wongwises, Ho Seon Ahn, and Omid Mahian. A hyper-optimisation method based on a physics-informed machine learning and point clouds for a flat plate solar collector. *Journal of Thermal Analysis and Calorimetry*, pp.

1–20, 2023. 2, 18 HC Hottel and ES Cohen. Radiant heat exchange in a gas-filled enclosure: Allowance for nonuniformity of gas temperature. *AIChE Journal*, 4(1):3–14, 1958. 16, 29 Zhili He, Futao Ni, Weiguo Wang, and Jian Zhang. A physics-informed deep learning method for solving direct and inverse heat conduction problems of materials. *Materials Today Communications*, 28:102719, 2021. 2, 18 Hoyt C Hottel and Adel F Saforim. *Radiative transfer*. McGraw-Hill, 1967. 16, 29 Yukun Hu, CK Tan, Jonathan Broughton, Paul Alun Roach, and Liz Varga. Model-based multiobjective optimisation of reheating furnace operations using genetic algorithm. *Energy Procedia*,
142:2143–2151, 2017. 2, 17 Yukun Hu, CK Tan, Jonathan Broughton, and Paul Alun Roach. Development of a first-principles hybrid model for large-scale reheating furnaces. *Applied Energy*, 173:555–566, 2016. 1, 4, 5, 16, 28, 29, 31 Yukun Hu, CK Tan, Jonathan Broughton, Paul Alun Roach, and Liz Varga. Nonlinear dynamic simulation and control of large-scale reheating furnace operations using a zone method based model. *Applied Thermal Engineering*, 135:41–53, 2018. 2, 17 Yukun Hu, CK Tan, John Niska, Jahedul Islam Chowdhury, Nazmiye Balta-Ozkan, Liz Varga, Paul Alun Roach, and Chunsheng Wang. Modelling and simulation of steel reheating processes under oxy-fuel combustion conditions–technical and environmental perspectives. *Energy*, 185:
730–743, 2019. 1, 7, 15, 25, 26, 28 Soonsung Hwang, Gunwoo Jeon, Jongpil Jeong, and JunYoul Lee. A novel time series based seq2seq model for temperature prediction in firing furnace process. *Procedia Computer Science*, 155:
19–26, 2019. 2, 18 IOM3 report. IOM3 report. https://www.iom3.org/resource/
transforming-foundations-industries.html, 2023. 1, 15 Jung Hyun Jang, Dong Eun Lee, Man Young Kim, and Hyong Gon Kim. Investigation of the slab heating characteristics in a reheating furnace with the formation and growth of scale on the slab surface. *International Journal of Heat and Mass Transfer*, 53(19-20):4326–4332, 2010. 2, 17 Gang Jing, Chenguang Ning, Jingwen Qin, Xudong Ding, Peiyong Duan, Haitao Liu, and Huiyun Sang. Physics-guided framework of neural network for fast full-field temperature prediction of indoor environment. *Journal of Building Engineering*, 68:106054, 2023. 2, 18 George Em Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang.

Physics-informed machine learning. *Nature Reviews Physics*, 3(6):422–440, 2021. 2, 18 Jong Gyu Kim and Kang Y Huh. Prediction of transient slab temperature distribution in the re-heating furnace of a walking-beam type for rolling of steel slabs. *ISIJ international*, 40(11):1115–1123, 2000. 2, 17 Kyung Mo Kim, Paul Hurley, and Juliana Pacheco Duarte. Physics-informed machine learning-aided framework for prediction of minimum film boiling temperature. International Journal of Heat and Mass Transfer, 191:122839, 2022. 2, 18 Man Young Kim. A heat transfer model for the analysis of transient heating of the slab in a direct-fired walking beam type reheating furnace. *International Journal of Heat and Mass Transfer*, 50(19-20): 3740–3748, 2007. 2, 17 Manu Lahariya, Farzaneh Karami, Chris Develder, and Guillaume Crevecoeur. Physics-informed lstm network for flexibility identification in evaporative cooling system. *IEEE Transactions on* Industrial Informatics, 19(2):1484–1494, 2022. 2, 18 Guojun Li, Wenchao Ji, Linyang Wei, and Zhi Yi. A novel fuel supplies scheme based on the retrieval solutions of the decoupled zone method for reheating furnace. International Communications in Heat and Mass Transfer, 141:106572, 2023. 2, 17 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Kang Li. Eng-genes: a new genetic modelling approach for nonlinear dynamic systems. IFAC
Proceedings Volumes, 38(1):162–167, 2005. 2, 17 Tian Liang, Shanshan Wang, Chunyang Lu, Nan Jiang, Wenqi Long, Min Zhang, and Ruiqin Zhang. Environmental impact evaluation of an iron and steel plant in china: Normalized data and direct/indirect contribution. *Journal of Cleaner Production*, 264:121697, 2020. 1, 15 Ying-Xin Liao, Jin-Hua She, and Min Wu. Integrated hybrid-pso and fuzzy-nn decoupling control for temperature of reheating furnace. *IEEE transactions on industrial electronics*, 56(7):2704–2714, 2009. 2, 17 Jan Marti, Andreas Haselbacher, and Aldo Steinfeld. A numerical investigation of gas-particle suspensions as heat transfer media for high-temperature concentrated solar power. International Journal of Heat and Mass Transfer, 90:1056–1070, 2015. 2, 16 AD Matthew, CK Tan, PA Roach, J Ward, J Broughton, and A Heeley. Calculation of the radiative heat-exchange areas in a large-scale furnace with the use of the monte carlo method. Journal of Engineering Physics and Thermophysics, 87(3):732–742, 2014. 3, 29 Matthieu Melot, Jean-Yves Trepanier, Ricardo Camarero, and Eddy Petro. Comparison of two ´
models for radiative heat transfer in high temperature thermal plasmas. Modelling and Simulation in Engineering, 2011, 2011. 2, 17 Christopher L Muhich, Brian D Ehrhart, Ibraheam Al-Shankiti, Barbara J Ward, Charles B Musgrave, and Alan W Weimer. A review and perspective of efficient hydrogen generation via solar thermal water splitting. *Wiley Interdisciplinary Reviews: Energy and Environment*, 5(3):261–287, 2016. 2 Net Zero by 2050. Net zero by 2050: A roadmap for the global energy sector. https://www.iea.

org/reports/net-zero-by-2050, 2021. 15 Xuan Manh Nguyen, Pedro Rodriguez-Ayerbe, F Lawayeb, Didier Dumur, and Alain Mouchette.

Temperature control of reheating furnace based on distributed model predictive control. In 2014 18th International Conference on System Theory, Control and Computing (ICSTCC), pp. 726–731.

IEEE, 2014. 2, 17 Tobias Oschmann and Harald Kruggel-Emden. A novel method for the calculation of particle heat conduction and resolved 3d wall heat transfer for the cfd/dem approach. *Powder Technology*, 338: 289–303, 2018. 16 Junho Park. *Hybrid Machine Learning and Physics-Based Modeling Approaches for Process Control* and Optimization. PhD thesis, Brigham Young University, 2022. 2, 18 Wei Qin, Zilong Zhuang, Yang Liu, and Jie Xu. Sustainable service oriented equipment maintenance management of steel enterprises using a two-stage optimization approach. Robotics and Computer- Integrated Manufacturing, 75:102311, 2022. 1, 15 Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A
deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational physics*, 378:686–707, 2019. 2, 18 Ling Shen, Zhipeng Chen, Xinyi Wang, and Jianjun He. Soft sensor modeling for 3d transient temperature field of large-scale aluminum alloy workpieces based on multi-loss consistency optimization pinn. *Sensors*, 23(14):6371, 2023. 2, 18 Guangwu Tang, Bin Wu, Dengqi Bai, Yufeng Wang, Rick Bodnar, and Chenn Q Zhou. Modeling of the slab heating process in a walking beam reheating furnace for process optimization. International Journal of Heat and Mass Transfer, 113:1142–1151, 2017. 2, 17 Josef Tausendschon and Stefan Radl. Deep neural network-based heat radiation modelling between ¨
particles and between walls and particles. *International Journal of Heat and Mass Transfer*, 177:
121557, 2021. 2, 17 Hong-Chuong Tran and Yu-Lung Lo. Heat transfer simulations of selective laser melting process based on volumetric heat source with powder size consideration. Journal of Materials Processing Technology, 255:411–425, 2018. 2 Ruihang Wang, Zhiwei Cao, Xin Zhou, Yonggang Wen, and Rui Tan. Phyllis: Physics-informed lifelong reinforcement learning for data center cooling control. In *Proceedings of the 14th ACM*
International Conference on Future Energy Systems, pp. 114–126, 2023. 2, 18 Gregor D Wehinger. Radiation matters in fixed-bed cfd simulations. *Chemie Ingenieur Technik*, 91
(5):583–591, 2019. 16 Mark D Wilkinson, Michel Dumontier, IJsbrand Jan Aalbersberg, Gabrielle Appleton, Myles Axton, Arie Baak, Niklas Blomberg, Jan-Willem Boiten, Luiz Bonino da Silva Santos, Philip E Bourne, et al. The fair guiding principles for scientific data management and stewardship. *Scientific data*, 3 (1):1–9, 2016. 33 648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701

## A Appendix

702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 Hong Yu, Jiangnan Gong, Guoyin Wang, and Xiaofang Chen. A hybrid model for billet tapping temperature prediction and optimization in reheating furnace. IEEE Transactions on Industrial Informatics, 2022. 2, 17 Qing-bo Yu, Zhong-wu Lu, and Jiu-ju Cai. Calculating method for influence of material flow on energy consumption in steel manufacturing process. Journal of Iron and Steel Research, International, 14(2):46–51, 2007. 1, 15 Walter W Yuen. Rad-nnet, a neural network based correlation developed for a realistic simulation of the non-gray radiative heat transfer effect in three-dimensional gas-particle mixtures. International Journal of Heat and Mass Transfer, 52(13-14):3159–3168, 2009. 2, 17 Walter W Yuen and Ezra E Takara. The zonal method: A practical solution method for radiative transfer in nonisothermal inhomogeneous media. *Annual review of heat transfer*, 8, 1997. 1, 3, 14, 17, 28, 29 Silvia Maria Zanoli, Crescenzo Pepe, and Lorenzo Orlietti. Multi-mode model predictive control approach for steel billets reheating furnaces. *Sensors*, 23(8):3966, 2023. 2, 17 Naiju Zhai and Xiaofeng Zhou. Temperature prediction of heating furnace based on deep transfer learning. *Sensors*, 20(17):4676, 2020. 2, 17 Naiju Zhai, Xiaofeng Zhou, Shuai Li, and Haibo Shi. Soft sensor model for billet temperature in multiple heating furnaces based on transfer learning. IEEE Transactions on Instrumentation and Measurement, 2023. 2, 17 Qi Zhang, Jin Xu, Yujie Wang, Ali Hasanbeigi, Wei Zhang, Hongyou Lu, and Marlene Arens.

Comprehensive assessment of energy conservation and co2 emissions mitigation in china's iron and steel industry based on dynamic material flows. *Applied Energy*, 209:251–265, 2018. URL https://doi.org/10.1016/j.apenergy.2017.10.084. 1, 15 Xingang Zhao, Koroush Shirvan, Robert K Salko, and Fengdi Guo. On the prediction of critical heat flux using a physics-informed machine learning-aided framework. *Applied Thermal Engineering*,
164:114540, 2020. 2, 18 Jianhua Zhou, Yuwen Zhang, and JK Chen. Numerical simulation of laser irradiation to a randomly packed bimodal powder bed. *International Journal of Heat and Mass Transfer*, 52(13-14):3137– 3146, 2009. 2

(a) (b) (c)
Figure 4: Convergence of PBMLP in training, considering: a) Supervised, b) EBV, and c) EBS terms.

## A.1 Motivation Of Our Work

Yuen & Takara (1997) in their study, have proved the elegance and superiority of the zone method over contemporary counterparts to model the physical phenomenon in high-temperature processes. In our work, we use the zone method towards a real-world application for the Foundation Industries (FIs), applied to reheating furnaces, due to the close and natural association/ relation of the zone-method 756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809

R-MSE tG vs. λ R-MSE tS fur vs. λ R-MSE tS obs vs. λ mMAPE fr vs. λ λ R-MSE tG
0 10 20 30 40 0 2 4 6 8 10 0 10 20 30 40 0 2 4 6 8

(b)
λ R-MSE
 tS obs
(a)
λ R-MSE tS fu r

(c)
λ mMAPE f r 0.001 0.01 0.1 0.5 1 1.5 10 0.001 0.01 0.1 0.5 1 1.5 10 0.001 0.01 0.1 0.5 1 1.5 10 0.001 0.01 0.1 0.5 1 1.5 10

(d)
Figure 5: Performance metrics against varying λebv = λebs = λ in PBMLP.
with the latter. Foundation Industries (FIs) constitute glass, metals, cement, ceramics, bulk chemicals, paper, steel, etc. and provide crucial, foundational materials for a diverse set of economically relevant industries: automobiles, machinery, construction, household appliances, chemicals, etc. FIs are heavy revenue and employment drivers, for instance, FIs in the United Kingdom (UK) economy are worth £52B (EPSRC report), employ 0.25 million people, and comprise over 7000 businesses (IOM3 report). The rapid acceleration in urbanization and industrialization over the decades has also led to improved building design and construction techniques. Great emphasis has been gradually placed on efficient heat generation, distribution, reduction, and optimized material usage. However, despite their economic significance, as depicted by the above statistics, the FIs leverage energy-intensive methods. This makes FIs major industrial polluters and the largest consumers of natural resources across the globe. For example, in the UK, they produce 28 million tonnes of materials per year, and generate 10% of the entire UK's CO2 emissions (EPSRC report; IOM3 report). Similarly, in China, the steel industry accounted for 15% of the total energy consumption, and 15.4% of the total CO2 emissions (Zhang et al., 2018; Liang et al., 2020). These numbers put a challenge for the FIs in meeting our commitment to reduce net Green-House Gas (GHG) emissions, globally.

Various approaches have been relied upon to achieve the Net-Zero trajectory in FIs (Net Zero by 2050): switching of grids to low carbon alternatives via green electricity, sustainable bio-fuel, and hydrogen sources, Carbon Capture and Storage (CCS), material reuse and recycling, etc. However, among all transformation enablers, a more proactive way to address the current challenges would be to tackle the core issue of process efficiency, via digitization, computer-integrated manufacturing, and control systems. Areas of impact by digitization could be reducing plant downtime, material and energy savings, resource efficiency, and industrial symbiosis, to name a few. Various computer-aided studies have already been conducted in notable industrial scenarios. The NSG Group's Pilkington UK Limited explored a sensor-driven Machine Learning (ML) model for product quality variation prediction (up to 72h), to reduce CO2 emission by 30% till 2030 (IOM3 report). Similar studies on service-oriented enterprise solutions for the steel industry have also been done recently in China (Qin et al., 2022). In this work, we tackle the key challenge of accurate and real-time temperature prediction in reheating furnaces, which are the energy-intensive bottlenecks common across the FIs. To give a perspective to the reader on why this is important, considering any process industry, such as the steel industry, one can observe that at the core, lies the process of conversion of materials (e.g., iron) into final products. This is done using a series of unit processes (Yu et al., 2007). The production process involves key steps such as dressing, sintering, smelting, casting, rolling, etc. A nice illustration of the different stages and processes in the steel industry can be found in Qin et al. (2022). The equipment in such process industries operates in high-intensity environments (e.g., high temperature), and has bottleneck components such as reheating furnaces, which require complex restart processes post-failure. This causes additional labor costs and energy consumption. Thus, for sustainable manufacturing, it is important to monitor the operating status of the furnaces via the furnace temperature profile. A few studies (Hu et al., 2019) have shown promise in achieving notable fuel consumption reduction by reducing the overall heating time by even as less as 13 minutes while employing alternate combustion fuels. A key area of improvement for furnace operating status monitoring lies in leveraging efficient computational temperature control mechanisms within them. This is because energy consumption per kilogram of CO2 could be reduced by a reduction in overall heating time.

As existing computational surrogate models have predictive capability bottlenecks, DL approaches can be used as suitable alternatives for real-time prediction. However, as only a handful of sensors/ thermo-couples could be physically placed within real-world furnaces (and that too at specific furnace 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 walls), the challenge of obtaining good-quality real-world data at scale to train DL models in such scenarios remains infeasible. To alleviate this, we identify the classical Hottel's zone method (Hottel & Cohen, 1958; Hottel & Saforim, 1967) which provides an elegant, iterative way to computationally model the temperature profile within a furnace, requiring only a few initial entities which are easily measurable. However, straightforward utilization of the same is not suitable for real-time deployment and prediction, due to computational expensiveness. For this reason, we propose that we generate an offline data set using the zone method, consisting of input-output pairs to train and evaluate ML models. We will provide a detailed description of the data generation methodology using the zone method.

## A.1.1 Computational Models

Available computational surrogate models based on Computational Fluid Dynamics (CFD) (Wehinger, 2019; De Beer et al., 2017), Discrete Element Method (DEM) (Emady et al., 2016), CFD-DEM hybrids (Oschmann & Kruggel-Emden, 2018), Two Fluid Models (TFM) (Marti et al., 2015), etc. incur expensive and time-consuming data acquisition, design, optimization, and high inference times. To break through the predictive capability bottlenecks of these surrogate models, DL approaches can be suitable candidates for real-time prediction, owing to their accuracy and inherently faster inference times (often only in the order of milliseconds).

## A.1.2 Discussion On Computational Aspects

In general, PINNs/ PCNNs and accurate simulators (e.g., CFD models) are two different approaches to solving a physical problem. In terms of computational efficiency, they cannot be compared at the same level. While PCNNs could take milliseconds for inference, accurate simulators have difficulty even achieving real-time simulation. Thus, PCNNs have the potential to be integrated directly into a control system for real-time control. This is because PCNNs are a type of approaches that encode the governing equations of the problem into the network training, whereas, accurate simulators are based on numerical methods that discretize the problem domain and solve the equations on a mesh, which can be time-consuming, and challenging to generate for complex geometries or moving boundaries (such as the furnace studied in our work). Generally speaking, the zone method is faster and simpler to implement than the CFD method. For example, even with a consumer-level PC, to simulate a 341-min real reheating process, the zone model only takes 5 mins, but CFD models often take several days, if not weeks, to provide *useful* results (Hu et al., 2016). Therefore, in this study, we utilize the zone model to generate training data for PCNNs. In future studies, the trained PCNNs will be integrated directly into furnace control systems. For our study, typically, generating 1500 timesteps of data for a single furnace using the zone method took about 2 hours, including the time for setting different configurations. However, talking about the absolute time of a CFD case simulation itself depends on many factors, such as mesh density, sub-model selection, step size settings, and computer hardware configuration.

Specific to our case, using the same configuration of PC, CFD simulation of the steady-state operating conditions of each setting takes about 5 hours. So the total time taken is 5 hours multiplied by the number of simulated working conditions. For the simulation of unsteady operating conditions, CFD is currently very difficult to implement, and some simplifications must be made. The specific time consumption depends on the duration of the simulated unsteady process. For the real process of 341 min for the case we studied, CFD would take at least 5 days (vs, 5 min of the zone method). As for the neural-network based implementations, for ML-based inference on a Apple M2 Max 32GB, our PCNN takes roughly 0.5s for inferring the entire furnace profile for a single time step instance, given the input variables as discussed.

## A.1.3 Computational Efficiency (Training And Testing Time) Between Methods With And Without Energy-Balance Based Physics-Regularization

The training time per mini-batch/iteration increases by up to 10x for smaller batch sizes when compared to the vanilla variant without Energy-Balance (EB) regularization. This increase is primarily due to the various matrix multiplications involving the DFA/TEA terms with higher-order matrices, particularly from the surface zones that comprise the regularization terms. However, when considering absolute run times, the increase is minimal; for example, the runtime per mini-batch is 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 approximately 76.11 seconds/iteration. We could reduce this further by using larger batch sizes to fully leverage GPU capabilities, although the performance gains would be marginal. In contrast, the simpler vanilla variants have a runtime of about 7.48 seconds/iteration. During inference, the time remains the same for both variants, as the regularization terms are only required during training for the Physics-Based (PB) variants, with no changes in the architectures.

## A.2 Details Of Related Work

While the research conducted in this work is at nascent stage, we believe it could pave way for further developments from an ML perspective, to solve a real-world application problem with value in terms of environmental sustainability. Our work, for an applied physical sciences reader, could inspire how ML and DL could be used to address a niche domain scenario. At the same time, for an ML audience, we believe that our work showcases a novel way to integrate physics based constraints into a neural network, especially using the zone method. Arguably, there exists a plethora of works related to PINNs, however, using PINNs to incorporate the zone method based regularizers as in our work, is a novel contribution to the community. The motivation to leverage the zone method also comes from the fact that it provides an elegant (and superior) way, as studied by Yuen & Takara (1997), to model the physical phenomenon in high-temperature processes inside reheating furnaces. In this section, we exhaustively present a set of relevant approaches with which our work can be loosely associated with. Specifically, we categorize them into two major classes: i) nonlinear dynamic systems, radiative heat transfer and view factor modeling, and, ii) modeling in reheating furnaces. We also talk about PINNs, and how our method is unique with respect to the existing literature.

(Category 1) Nonlinear dynamic systems, radiative heat transfer and view factor modeling:
Our work at its heart is based on the zone method, which in turn relies on notions of radiative heat transfer and view factor modeling (or interchangeably, exchange area calculation). Describing the behavior of a furnace state involves combustion models, control loops, set point calculations, and fuel flux control in zones. It also involves linearization and model order reduction for state estimation and state-space control. The inherent complexity makes the modeling a nonlinear dynamic system.

While there is no exact similarity, our work shares some common philosophies with few earlier works. For instance, Ebrahimi et al. (2013) discuss the modeling of radiative heat transfer using simplified exchange area calculation. Radiative heat transfer in high-temperature thermal plasmas has been studied by Melot et al. (2011) while comparing two models. A nonlinear dynamic simulation and control based method has been studied by Hu et al. (2018). A classical work based on genetic algorithm for nonlinear dynamic systems (Li, 2005) is also present, which, instead of a data-driven approach, leverages a pre-defined set of mathematical functions. Within this category, some approaches have also employed neural networks. In Yuen (2009), a network was trained for simulating non-gray radiative heat transfer effect in 3D gas-particle mixtures. Some approaches have used networks for view factor modeling with DEM-based simulations (Tausendschon¨ & Radl, 2021), and some have addressed the near-field heat transfer or close regime (Garc´ıa-Esteban et al., 2021). (Category 2) Modeling in reheating furnaces: We now discuss methods dealing with some form of prediction or optimization in reheating furnaces. Classically, Kim & Huh (2000) discussed a method to predict transient slab temperatures in a walking-beam furnace for rolling of steel slabs. Kim (2007) proposed a model for analyzing transient slab heating in a direct-fired walking beam furnace. Jang et al. (2010) investigated the slab heating characteristics with the formation and growth of scale. Tang et al. (2017) studied slab heating for process optimization. A distributed model predictive control approach was proposed in Nguyen et al. (2014). Few multi-objective optimization methods were discussed in Hu et al. (2017); Ban et al. (2023). A fuel supplies scheme based approach was proposed in Li et al. (2023). Other related works involved multi-mode model predictive control approach for steel billets (Zanoli et al., 2023), and a hybrid model for billet tapping temperature prediction (Yu et al., 2022). Some neural network based approaches in this category studied transfer learning (Zhai & Zhou, 2020; Zhai et al., 2023), digital twin modeling (Halme Stahlberg ˚ , 2021), and steel slab temperature prediction (de Souza Lima et al., 2023). Liao et al. (2009) discussed an integrated hybrid-PSO and fuzzy-NN decoupling based solution. Other works have studied aspects related to time-series 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 modeling (Hwang et al., 2019; Chen et al., 2022), and multivariate linear-regression in steel rolling (Bao et al., 2023). PINNs: The methods mentioned above discuss alternatives aimed at modeling either exchange factors with radiative heat transfer, or specific slab temperature predictions in reheating furnaces. However, they do not explicitly address physics-based prior incorporation within their optimization frameworks, especially for the neural network variants. To this end, we now discuss a few relevant works in the body of literature on PINNs. For a detailed review on PINNs in general, we refer the interested reader to the papers by Raissi et al. (2019); Karniadakis et al. (2021). It should be noted that PINNs are a broad category of approaches, and the literature is vast. Here, we discuss those methods which relate to certain aspects of thermal modeling. Drgona et al. ˇ (2021) proposed a physics-constrained method to model multi-zone building thermal dynamics. A multi-loss consistency optimization PINN (Shen et al., 2023) was proposed for largescale aluminium alloy workpieces. Other approaches focus on prototype heat transfer problems and power electronics applications Cai et al. (2021), minimum film boiling temperature (Kim et al., 2022), critical heat flux (Zhao et al., 2020), solving direct and inverse heat conduction problems of materials (He et al., 2021), lifelong learning in district heating systems (Boca de Giuli, 2023), PINN and point clouds for flat plate solar collector (Han et al., 2023), residential building MPC (Bunning et al. ¨ , 2022),
hybrid ML and PINN for Process Control and Optimization (Park, 2022), reinforcement learning for data center cooling control (Wang et al., 2023), flexibility identification in evaporative cooling (Lahariya et al., 2022), and fast full-field temperature prediction of indoor environment (Jing et al., 2023). Uniqueness of our work within existing literature: While we have observed a number of loosely related methods as discussed above, upon a clear look at them, we can conclude the following:
1. **Comparison with category 1 methods:** Among the approaches focusing on view factor modeling with radiative transfer, the area of interest is often simplified. The modeling covers select few exchange areas. The methods are also geometry-specific. Our approach on the other hand seeks a generic, geometry-agnostic modeling that covers the entire set of exchange areas. The exchange areas can be intuitively perceived as those interfaces from where radiation can transfer, between a pair of zones (surface/gas). A background on exchange areas is provided in the proposed work section. The ones involving neural networks, often employ feed-forward Multi-Layer Perceptron (MLP) models with few hidden layers. As showcased in our experiments, a simple MLP trained to regress the outputs given certain inputs may not generalize well to unseen distributions, due to lack of explicit understanding of the underlying physics. On the other hand, we empirically showcase that our proposed PCNN performs better than such a baseline MLP.

Within a single PCNN framework, our method can also cover other architectures such as LSTMs, KANs, xLSTMs etc.

2. **Comparison with category 2 methods:** Both non-neural and neural-network based methods presented in this category, as observed, focus on predicting temperatures only in certain regions of a furnace, often, the slab temperature profiling. Our work, on the other hand aims at achieving a complete furnace temperature profiling, ranging from the gas zones, to both types of surface zones: furnace walls as well as the slab/obstacle surfaces.

Our training data set is obtained based on the iterative zone method, and is more holistic in nature as compared to the discussed methods. This makes an apple-to-apple comparison difficult with other methods as they deal with different problem setups. Furthermore, the neural methods in this category are not trained to be physics aware.

3. **Comparison with PINNs:** It should be noted that any PINN approach is driven by the priors corresponding to the underlying physical phenomenon. As we did not find PINN methods addressing zone method based modeling, we could claim our PCNN variant to be novel in nature, especially, in this studied problem setup. Essentially, casting the temperature prediction task in reheating furnaces as in our work, and modeling via explicit physicsconstrained regularizers (based on zone method) as done in our work, is a first of its kind. It is a simple paradigm, and could be used to build further sophisticated developments. At the same time, it simply requires input-output pairs (as shown later) to train the underlying ML/PCNN model, and makes no geometry-specific assumptions of the furnace. The data creation method discussed in our method is holistic, covers all possible exchange areas, and thus, is unique in nature itself.

## A.3 Performance Metrics

For a data set containing N samples: X = {(x
(i), y
(i))}
N
i=1, we make use of the following standard regression performance evaluation metrics:
1. **Root Mean Squared Error (RMSE)**, defined as:
972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025

$$M A E={\frac{\sum_{i=1}^{N}\left|\mathbf{y}^{(i)}-f_{\theta}(\mathbf{x}^{(i)})\right|}{N}}$$

N(14)
Mean Absolute Percentage Error (MAPE) is unsuitable for firing rate prediction due to potential division by zero. We use a modified MAPE (mMAPE) with a small epsilon (ϵ = 0.05) added to the denominator:

$$m M A P E={\frac{1}{N}}\sum_{t=1}^{N}\left|{\frac{f_{t}-{\hat{f}}_{t}}{f_{t}+\epsilon}}\right|\,$$
Here, ft is the actual firing rate, and ˆft is the predicted value.
$$(15)$$
$$R M S E={\sqrt{\frac{\sum_{i=1}^{N}(\mathbf{y}^{(i)}-f_{\theta}(\mathbf{x}^{(i)}))^{2}}{N}}}$$
$$(13)$$

$$(14)$$

2. **Mean Absolute Error (MAE)**, defined as:
We evaluate model performance for each entity (gas zone temperatures, tG; furnace surface temperatures, tS fur; obstacle surface temperatures, tS obs; firing rates, fr) separately as: RMSE tG, RMSE tS fur, RMSE tS obs, MAE tG, MAE tS fur, MAE tS obs, and mMAPE fr. Performance metrics
(RMSE, MAE, mMAPE) are computed using corresponding predictions from the model (fθ(x
(i)))
and ground truth values from the data (y
(i)). Results are presented for the test split (standard practice).

mMAPE is evaluated only for the firing rates. RMSE, MAE and mMAPE range in [0, ∞] with lower values indicating better performance (↓) as shown in the tables.

## A.4 Training Details And Model Architectures

We train our PBMLP for 10 epochs using PyTorch (early stopping to avoid over-fitting), and report results with the final checkpoint. For the EB equations, we perform the same normalization for enthalpy, flux, and temperatures, as in the final neural network output as discussed earlier. We found a learning rate of 0.001 with Adam optimizer and batch size of 64 to be optimal, along with ReLU non-linearity. We pick the [50,100,200] configuration for hidden layers, i.e., 3 hidden layers, with 50, 100, and 200 neurons respectively. We use λebv = λebs = 0.1. In general, a value lesser than 1 is observed to be better, otherwise, the model focuses less on the regression task. Following are values of other variables: |G| = 24, |S| = 178 (76 furnace surface zones and 102 obstacle surface zones), Ng = 6, and Stefan-Boltzmann constant=5.6687e-08. Unless otherwise stated, this is the setting we use to report any results for our method, for example, while comparing with other methods. Please note that the MLP baseline has exactly the same training configuration as the PBMLP except that it does not use the physics regularizers. We provide details about the LSTM variants used. The LSTM variant has a single LSTM layer with 50 hidden nodes, followed by FC layer-1 with 50 input nodes and 100 output nodes, FC layer-2 with 100 input nodes and 200 output nodes. Both FC layer-1 and FC layer-2 have ReLU non-linearity.

Lastly, there is a final FC layer with sigmoid nonlinearity that maps to the number of output features as in the data set. The DLSTM variant has three stacked LSTM layers, each with 100 hidden nodes, followed by a final FC layer with sigmoid nonlinearity. As we can see, we have kept the total number of layers in LSTM and DSLTM comparable to that of the baseline MLP.

For the xLSTM implementation, we follow a similar architeture as the DLSTM model. Similar to the DLSTM we place a LSTM layer that maps the input to 100 hidden nodes. However, after that, instead of stacking two more LSTM layers, we place a single xLSTM block stack (as mentioned in the official repository https://github.com/NX-AI/xlstm). After the xLSTM block, the remaining layers are similar to that of the DLSTM. Within the xLSTM block stack, the sLSTM block has 4 heads, 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 conv1d kernel size=4, and, the mLSTM block has conv1d kernel size=4, qkv proj blocksize=4, and 4 heads. Overall, xLSTM block has context length of 1, 7 blocks, and embedding dimension of 100. For KAN, we follow the implementation suggestions as in https://github.com/ KindXiaoming/pykan and use a single hidden layer with one neuron. Interestingly, the KAN despite being simpler than the MLP baseline, is not only easier to train, but also outperforms the MLP, as evidenced in many contemporary works. Broadly speaking, the training specific hyperparameters across all the compared models are the same (e.g., number of epochs, optimizer, batch size, learning rate, etc). The only difference comes from their respective architectures. For a similar architecture, the additional difference for the physics based variants lie in terms of usage of the additional regularization terms. Table 5 summarizes the details.

| Table 5: Architectural and training details across different studied models   |                                                                                                                 |                                            |
|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| Model                                                                         | Architecture                                                                                                    | Layer-specific information                 |
| MLP                                                                           | 3 hidden layers (50, 100, 200 neurons)+ Final FC layer (no. of outputs)                                         | -                                          |
| LSTM                                                                          | 1 LSTM layer (50 hidden nodes) + 2 FC layers (FC-1 and FC-2) + Final FC layer (no. of outputs)                  | FC-1: 50-100, FC-2: 100-200                |
| DLSTM                                                                         | 3 stacked LSTM layers (100 hidden nodes each) + Final FC layer (no. of outputs)                                 | -                                          |
| xLSTM block: context length = 1, #blocks =7, embedding dim = 100              |                                                                                                                 |                                            |
| xLSTM                                                                         | 1 LSTM layer (100 hidden nodes) +                                                                               | sLSTM block:#heads=4, conv1d kernel size=4 |
| 1 xLSTM block + Final FC layer (no. of outputs)                               | mLSTM block: #heads=4,                                                                                          |                                            |
| conv1d kernel size=4, qkv proj blocksize=4                                    |                                                                                                                 |                                            |
| KAN                                                                           | 1 hidden layer (1 neuron)+ Final FC layer (no. of outputs)                                                      | -                                          |
| PB-variants                                                                   | Same as corresponding base architecture, but additionally use physics-based regularizers with λebv = λebs = 0.1 |                                            |
| Common Hyperparameters: 10 epochs, Adam optimizer, lr=0.001, batch size=64    |                                                                                                                 |                                            |

| Table 6: All results (Normal Type 1 Datasets)   |                                                             |                   |      |      |       |      |      |      |       |      |
|-------------------------------------------------|-------------------------------------------------------------|-------------------|------|------|-------|------|------|------|-------|------|
| Dataset                                         | N1-1                                                        | 925 1220 1250 750 |      |      |       |      |      |      |       |      |
| Metric/ Method                                  | MLP PBMLP LSTM PBLSTM DLSTM PBDLSTM KAN PBKAN xLSTM PBxLSTM |                   |      |      |       |      |      |      |       |      |
| RMSE tG (↓)                                     | 136.4                                                       | 55.3              | 15.6 | 43.3 | 28.4  | 16.1 | 40.7 | 12.6 | 39.6  | 13.7 |
| RMSE tS fur (↓)                                 | 139.2                                                       | 39.8              | 7.1  | 39.3 | 13.8  | 6.3  | 34.4 | 9.7  | 38.3  | 10.6 |
| RMSE tS obs (↓)                                 | 124.8                                                       | 64.9              | 43.7 | 73.8 | 54.2  | 52.6 | 54.2 | 21.2 | 63.9  | 22.8 |
| MAE tG (↓)                                      | 108.6                                                       | 51.0              | 11.1 | 39.5 | 20.7  | 10.9 | 38.8 | 10.2 | 37.5  | 11.7 |
| MAE tS fur (↓)                                  | 115.7                                                       | 39.2              | 6.0  | 38.1 | 12.2  | 5.1  | 34.1 | 9.1  | 37.8  | 10.0 |
| MAE tS obs (↓)                                  | 100.2                                                       | 54.8              | 19.5 | 58.1 | 32.1  | 22.1 | 50.1 | 18.1 | 59.3  | 18.7 |
| mMAPE fr (↓)                                    | 232.9                                                       | 70.7              | 25.6 | 26.5 | 21.9  | 23.7 | 51.1 | 40.7 | 22.1  | 27.6 |
| Dataset                                         | N1-2                                                        | 965 1220 1250 750 |      |      |       |      |      |      |       |      |
| Metric/ Method                                  | MLP PBMLP LSTM PBLSTM DLSTM PBDLSTM KAN PBKAN xLSTM PBxLSTM |                   |      |      |       |      |      |      |       |      |
| RMSE tG (↓)                                     | 113.4                                                       | 35.6              | 33.0 | 26.7 | 117.1 | 32.4 | 24.3 | 22.6 | 130.6 | 29.3 |
| RMSE tS fur (↓)                                 | 116.4                                                       | 22.4              | 25.6 | 11.7 | 114.4 | 24.9 | 15.2 | 14.6 | 119.1 | 20.4 |
| RMSE tS obs (↓)                                 | 106.9                                                       | 43.4              | 61.1 | 66.5 | 109.3 | 67.4 | 35.1 | 33.6 | 139.8 | 45.4 |
| MAE tG (↓)                                      | 89.5                                                        | 28.2              | 27.4 | 16.9 | 100.9 | 27.2 | 21.4 | 19.9 | 129.1 | 26.8 |
| MAE tS fur (↓)                                  | 96.2                                                        | 17.8              | 21.5 | 9.9  | 101.1 | 20.1 | 14.3 | 13.8 | 118.6 | 19.5 |
| MAE tS obs (↓)                                  | 79.9                                                        | 29.6              | 39.4 | 31.4 | 86.9  | 44.4 | 29.8 | 29.3 | 136.3 | 39.8 |
| mMAPE fr (↓)                                    | 176.6                                                       | 58.5              | 29.5 | 23.5 | 201.0 | 26.2 | 44.2 | 32.6 | 200.8 | 27.8 |
| Dataset                                         | N1-3                                                        | 995 1220 1250 750 |      |      |       |      |      |      |       |      |
| Metric/ Method                                  | MLP PBMLP LSTM PBLSTM DLSTM PBDLSTM KAN PBKAN xLSTM PBxLSTM |                   |      |      |       |      |      |      |       |      |
| RMSE tG (↓)                                     | 31.1                                                        | 30.5              | 39.3 | 39.2 | 100.0 | 35.7 | 23.1 | 20.9 | 114.9 | 30.1 |
| RMSE tS fur (↓)                                 | 22.1                                                        | 24.3              | 8.0  | 16.5 | 97.0  | 25.8 | 18.4 | 17.1 | 104.3 | 23.1 |
| RMSE tS obs (↓)                                 | 54.4                                                        | 47.8              | 69.0 | 77.4 | 97.2  | 60.5 | 27.7 | 26.4 | 124.2 | 35.1 |
| MAE tG (↓)                                      | 23.0                                                        | 23.8              | 25.3 | 29.1 | 87.0  | 29.4 | 20.9 | 18.4 | 113.6 | 27.9 |
| MAE tS fur (↓)                                  | 16.8                                                        | 20.8              | 6.4  | 14.6 | 85.8  | 22.4 | 17.7 | 16.4 | 104.1 | 22.4 |
| MAE tS obs (↓)                                  | 31.4                                                        | 29.4              | 36.6 | 46.5 | 73.1  | 32.7 | 24.0 | 22.5 | 120.7 | 30.4 |
| mMAPE fr (↓)                                    | 32.0                                                        | 28.1              | 25.8 | 26.9 | 128.7 | 29.4 | 33.0 | 27.7 | 127.7 | 31.7 |

## A.5 Full Set Of Results On The 11 Datasets

In Tables 6, 7, 8, 9 we report the performances of the compared approaches across all the 11 datasets. We noticed that not only the PB variants obtain a better performance throughout, they are also more stable across different datasets as indicated by their standard deviations (Table 10). On the other hand, the performances of the vanilla networks were not stable across different datasets.

However, we also noted that Physics-Based (PB) variants perform *slightly worse* than the vanilla methods in certain datasets. This because we did not tune hyperparameters for each configuration, but rather aimed to obtain average performance across configurations. While there may be potential for further improvements at the configuration level, our primary goal was to assess the generalizability of our approach. In real-world scenarios, variability is to be expected. It is possible that, for certain 20