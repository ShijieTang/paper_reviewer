000
001
002
003
004
005
006
007
008
009
010
011
012
013
014
015
016
017
018
019
020
021
022
023
024
025
026
027
028
029
030
031
032
033
034
035
036
037
038
039
040
041
042
043
044
045
046
047
048
049
050
051
052
053
054

# **The Expressivity of Fixed-Precision Transformers without Positional Encoding**

**Anonymous Authors** [1]

**Abstract**

The primary objective of this study is to examine
how practical constraints impact the expressivity
of Transformers and to investigate their expressivity in real-world implementations.

To achieve this, we analyze the expressivity
of Transformer decoders operating under fixedprecision float arithmetic, an assumption regarding query-key parameters, and the presence or absence of positional encoding. Our findings reveal
that, under fixed-precision and these constraints,
Transformers are limited to recognizing finite or
co-finite languages, a proper subclass of regular
languages. While incorporating positional encoding or relaxing certain assumptions marginally enhances expressivity, the fundamental limitations
imposed by fixed precision remain significant.

These results underscore the gap between theoretical models and real-world implementations,
suggesting that practical Transformers may be
fundamentally constrained to recognizing only
finite and co-finite languages, effectively functioning as little more than efficient lookup tables.

**1. Introduction**

The expressivity of Transformer models (Vaswani et al.,
2017) has been further elucidated through recent theoretical
analyses by comparing to the range of recognizable formal
languages and solvable complexity classes. A series of
studies has established upper and lower bounds on their
expressivity under following settings.

Perez et al.´ (2021) is the first study to explore the expressivity of Transformers, proving their Turing-completeness
using rational numbers, assuming infinite precision float.
Subsequent studies adopting finite precision have provided
more practical insights. For instance, Merrill & Sabharwal

1Anonymous Institution, Anonymous City, Anonymous Region,
Anonymous Country. Correspondence to: Anonymous Author
_<_ anon.email@domain.com _>_ .

Preliminary work. Under review by the International Conference
on Machine Learning (ICML). Do not distribute.

|Asm.|Assumption. 5.1|–|
|---|---|---|
|PE|NoPE<br>APE|NoPE<br>APE|
|Uppoer<br>bound|**FinCofn**<br>(§ 5.1)<br>?|?<br>FOC<br>[+; MOD]_†_|
|Lower<br>bound|**FinCofn**<br>(§ 5.2)<br>**FinCofn**<br>_m_**-cyclic**<br>(§ 6.1)|**FinCofn**<br>**letter-set**<br>(§ 6.2)<br>?|

(2023; 2024a) investigated logarithmic precision, which is
finite but scales with input length _n_, and revealed that such
Transformers are limited to much smaller circuit complexity
classes, such as TC [0] or logical class FO(M), compared to
Turing machines. Similarly, Chiang et al. (2023) examined
fixed-precision Transformers and demonstrated that their
tighter upper bounds are linked to logic FOC[+; MOD],
which is an extension of first-order logic.

Despite these theoretical advances, many studies rely on idealized conditions. This paper bridges the gap between these
settings and real-world implementations, which impose significant constraints on processing and retaining information.

We investigate how the expressivity of Transformer decoders is shaped by the following practical constraints:
_fixed-precision_ _floating-point_ _numbers_, _positional_ _encod-_
_ing variations_ (APE, NoPE), and _assumptions on parameter_
_configurations_ (asm. 5.1). Our results indicate that expressivity depends on these constraints as follows (Table 1).

  - Fixed-precision (e.g., fp32, bf16) limits recognition to
finite and co-finite languages. _{a, b, ba, aab}_ (§ 5)

  - Absolute positional encoding extends recognition beyond finite and co-finite languages to cyclic languages.
_{ab, abab, ababab, . . . }_ (§ 6.1)

  - Non-finite values ( _±_ inf) expand expressivity to letterset languages, capturing specific letter inclusion.
_{abbb, ccac, bbac, . . . }_ (§ 6.2)

Our findings extend prior results by highlighting the theoretical and practical implications of fixed-precision.

_Table 1._ The upper and lower bound of the expressivity of fixedprecision Transformers. Chiang et al. (2023) _[†]_ identified the upper
bound of normal Transformer encoder model (i.e. fixed-precision,
sinusoidal positinoal encoding). In this study we showed that **bold**
parts. “?” means the bound is not known.

1

055
056
057
058
059
060
061
062
063
064
065
066
067
068
069
070
071
072
073
074
075
076
077
078
079
080
081
082
083
084
085
086
087
088
089
090
091
092
093
094
095
096
097
098
099
100
101
102
103
104
105
106
107
108
109

**2. Related Work**

**2.1. Transformer Models and Expressivity**

The computational capabilities of neural networks, covering RNNs, CNNs, and Transformers, have been extensively
studied. The comprehensive surveys by Ackerman & Cybenko (2020); Merrill (2021; 2023) provide an in-depth
overview of the expressivity of neural networks as a whole.

Learnability is inherently bounded by expressivity, as the
language that a model can recognize defines the boundaries
of what it can effectively learn. Therefore, expressivity is
not only a theoretical concern, but it is also of practical
importance in guiding model design.

A survey paper (Strobl et al., 2024) and lecture notes (Chiang et al., 2024) provide a comprehensive overview of recent
advances in the study of Transformer expressivity, highlighting that expressivity is often analyzed in relation to three
key areas: formal languages, circuit complexity, and logic.

**Formal** **languages** Hahn (2020); Bhattamishra et al.
(2020a;b); Yao et al. (2021); Chiang & Cholak (2022) primarily investigated the relationship between variants of hardTransformers and formal languages such as PARITY and
Dyck languages, which are commonly used benchmarks
for expressivity. Feng et al. (2023); Merrill & Sabharwal
(2024b) focused on the decoding time, inspired by chainof-thought reasoning (Wei et al., 2022), demonstrating that
expressivity expands significantly with multiple decoding
steps. Of particular interest, Nowak et al. (2024) examined
how Transformers assign probabilities to strings in language
modeling, identifying connections to probabilistic deterministic finite automata and probabilistic Turing machines.

**Circuit complexity** Another perspective comes from circuit complexity theory, which classifies computational problems based on their implementability within Boolean circuits of bounded depth and size. Hao et al. (2022) analyzed
hard-Transformer variants, linking them to the tiny circuit
class AC [0] . Merrill et al. (2022); Merrill & Sabharwal (2023)
extended this to more practical settings, showing that the
saturated attention and logarithmic precision Transformers
remain within TC [0] . Merrill & Sabharwal (2023) further
suggested a fundamental parallelism trade-off, arguing that
highly parallel architectures like Transformers may inherently face computational limits.

**Logic** Chiang et al. (2023); Merrill & Sabharwal (2024a);
Yang et al. (2024); Yang & Chiang (2024) have explored
connections between Transformer models and first-order
logic. These studies encode strings into Boolean variables
and represent languages using logical frameworks such as
first-order logic with counting quantifiers (FOC[+; MOD]).

While significant progress has been made, many studies
rely on unrealistic assumptions such as infinite precision
or hard-attention, leaving questions about their practical
relevance.

**2.2. Neural Networks and Function Approximation**

A fundamental result in neural network theory is the universal approximation theorem, which states that any continuous
function can be approximated arbitrarily well. While not
the focus of our study, it provides essential context for understanding the broader capabilities of neural networks.

**Feedforward** **networks** Feedforward neural networks
(FFNs) play a central role in this context. Cybenko (1989);
Hornik et al. (1989) proved that FFNs with a single hidden layer and arbitrary nonlinear activations can universally
approximate any Borel measurable or continuous function,
given sufficient hidden units. Park et al. (2020) further
identified the minimum width required for universal approximation, given the input and output dimensions.

**Transformers** Recent work has extended universal approximation results to Transformers, with Yun et al.
(2020) establishing their ability to approximate continuous sequence-to-sequence functions on compact domains
and highlighting the crucial role of positional encoding in
encoding order and circumventing permutation equivariance constraints. Kajitsuka & Sato (2024) later showed that
even single-layer Transformers with low-rank weights can
achieve such approximation power. Furthermore, Wei et al.
(2022) introduced the statistically meaningful approximation framework, addressing limitations in classical approximation theory by incorporating learnability constraints.

**3. Preliminaries**

In this section, we present the foundational concepts that
support our theoretical results. For strings _w, w_ _[′]_ _∈_ Σ _[∗]_ over
the alphabet Σ, _|w|_ denotes the length of the string, and _ww_ _[′]_

denotes the concatenation. Furthermore, _wt_ denotes the _t_ -th
character, and _wi_ : _j_ ( _i, j_ _∈_ N) denotes the subsequence of _w_
from the _i_ -th to the _j_ -th character.

**3.1. Finite and Co-finite Languages, Cyclic Language,**
**Letter-set Language**

This subsection introduces _finite languages_ and their dual,
_co-finite_ _languages_, along with _letter-set_ _languages_ and
_cyclic languages_ . These languages will play a central role
in analyzing the expressivity of Transformers (§ 5, § 6).

**Definition 3.1** (Finite Language) **.** Let Σ be a finite alphabet.
A language _L ⊆_ Σ _[∗]_ is called a _finite language_ if and only if
there exists _k_ _∈_ N such that for all stings _w_ _∈_ _L, |w| ≤_ _k_ .

2

110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164

**Definition 3.2** (Co-finite Language) **.** Let Σ be a finite alphabet. A language _L ⊆_ Σ _[∗]_ is called a _co-finite language_ if
and only if its complement Σ _[∗]_ _\ L_ is a finite language.

**Definition 3.3** ( _m_ -cyclic Language) **.** Let Σ be a finite alphabet. A language _L ⊆_ Σ _[∗]_ is called a _m_ - _cyclic language_
if and only if for some _m ∈_ N, for all _w, w_ _[′]_ _∈_ _L_ and for all
0 _≤_ _i ≤_ max( _|w|, |w_ _[′]_ _|_ ), _wi_ _≡_ _wi_ _[′]_ [mod] _[m]_ [ holds.]

**Definition 3.4** (Letter-set Language) **.** Let Σ be a finite alphabet. A language _L ⊆_ Σ _[∗]_ is called a _letter-set language_
if and only if for some set of letters _A ⊆_ Σ, for all _w_ _∈_ _L_
includes all of the letters in _A_ .

**Example** **3.5.** _The_ _following_ _languages_ _L, L_ _[′]_ _over_ Σ =
_{a, b} are co-finite languages:_

_L_ = Σ _[∗]_ _\ {a, b, ab, aab}_

_L_ _[′]_ = _{w_ _∈_ Σ _[∗]_ _| |w| ≥_ 3 _}_

_Similarly, the following language L{a,b}_ _over_ Σ = _{a, b, c}_
_is letter-set language and L_ 3 _is_ 3 _-cyclic language:_

_L_ 3 = ( _abc_ ) _[∗]_

_L{a,b}_ = _{w_ _∈_ Σ _[∗]_ _| w_ has both _a_ and _b}_

**3.2.** _p_ **-precision Float-Point Numbers**

Now we define the rigorous mathematical framework for
representing and manipulating numerical values under finite precision constraints, following (Merrill & Sabharwal,
2023).

**Definition 3.6** ( _p_ -precision Floating-Point Numbers (Merrill
& Sabharwal, 2023)) **.** The set of _p_ -precision floating-point
numbers D _p_ is defined as the collection of _p_ -bit numbers,
D _p_ = _{_ 0 _,_ 1 _}_ _[p]_, including special values such as +inf, _−_ inf,
and nan. The set D _p_ can be naturally extended to vectors
_∗_
D _p_ .

When _p_ happens to be a finite number, we can also define
the operations over _p_ -precision float since the cardinality of
the mappings between D _p_ vectors of _m_ -dimension become
at most finite (= 2 _[pm][·]_ [2] _[pm]_ ).

**Definition 3.7** (Merrill & Sabharwal (2023)) **.** A function
_m_ _n_
_f_ : D _p_ _→_ D _p_ is a _p_ -precision floating-point function if
_f_ can be computed by a _p_ -space-bounded Turing machine.

The order and basic operations, including addition, subtraction, multiplication, and division, as well as operations
involving special values (+inf, _−_ inf, nan), follow the IEEE
754 standard (iee, 2019).

The precision _p_ can be defined as a function of the input
sequence length _n_, determining the scale of precision as
follows:

  - Constant Precision: When _p_ ( _n_ ) is a constant function
( _p_ ( _n_ ) _∈O_ (1)), the precision is fixed for any length of
input.

  - Logarithmic Precision: When _p_ ( _n_ ) is a logarithmic
function ( _p_ ( _n_ ) _∈O_ (log _n_ )), the precision scales logarithmically with the input length.

In this work, our concern is constant precision. In the case
of constant precision, _p_ can be treated as a constant _p ∈_ N.

**4. Transformer Decoder**

This section introduces the mathematical and theoretic foundations of the Transformer Decoder Model, emphasizing its
functional behavior (Def. 4.1), autoregressive capabilities
(Def. 4.2), and alignment with formal languages (Def. 4.3).

**4.1. Transformer Decoder**

We focus on the decoder-based GPT (generative pretrained
transformer) architecture (Radford et al., 2018). Unlike the
original implementation, positional encoding (PE; details
in § 4.1) is excluded in § 5 to facilitate theoretical analysis,
while it is included in § 6 to reflect practical settings and
evaluate its impact.

In this work, all computations within the Transformer are
conducted over the _p_ -precision float numbers D _p_ (see § 3.2
and Merrill & Sabharwal (2023)). This constraint reflects a
practical adaptation to real-world computational limits.

**Vocabulary space** The vocabulary space of Transformers
Σ _∪_ V comprises the alphabet Σ and a set of special tokens V
(e.g., _⟨_ bos _⟩_, _⟨_ eos _⟩_, _⟨_ sep _⟩_ ). Special tokens are excluded from
formal language, and there is no intersection to alphabet in
this study. Basic string operations, such as concatenation,
closure, and length, are defined over vocabulary spaces in
the standard manner.

**Transformer as a function** Then we formalize the Transformer Decoder model as a function.

**Definition 4.1.** A Transformer Decoder over _p_ ( _n_ )-precision
with parameters _θ_ _∈_ Params is a function:

TDec _p_ ( _·_ ; _θ_ ) : (Σ _∪_ V) _[∗]_ _→_ Σ _∪_ V

where Σ _∪_ V is the vocabulary space. Params represents
the class of trainable parameters set, all components of the
model. _p_ ( _n_ ) determines the internal precision depend on
the input sequence length _n_ .

Given an input sequence _w_ 1: _n_ _∈_ (Σ _∪_ V) _[∗]_, the Transformer Decoder outputs a single next token _wn_ +1 =
TDec _p_ ( _n_ )( _w_ 1: _n_ ; _θ_ ) _∈_ Σ _∪_ V, conditioned on the prefix _w_ 1: _n_
and a set of parameter _θ_ . Based on the formal definition of
TDec, the computational flow from input to output generally follows the GPT model (Radford et al., 2018; Brown
et al., 2020).

3

165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219

**Positional encoding** Since encoder-only Transformer cannot recognize the position of character, they need additional
positional information. We denote abusolute positional encoding (APE) like Vaswani et al. (2017)’s sinusoidal one in
this work. On the other hand, there are relative ones like
T5 relative PE (Raffel et al., 2020) or ALiBi (Press et al.,
2022). Alternatively, Kazemnejad et al. (2024) showed _No_
_Positional Encoding (NoPE)_ has good ability to generalize.
In this work, we employ only APE and NoPE.

**4.2. Autoregressive Token Generation**

The Transformer Decoder model generates sentences autoregressively, predicting each token based on the input
sequence and previously generated tokens until output a
kind of end-of-sentence tokens. This process is formalized
as follows:

**Definition** **4.2.** The _t_ -times autoregressive composition (generation) of the Transformer Decoder function
TDec _p_ ( _·_ ; _θ_ ) is denoted as TDec _pt_ ( _·_ ; _θ_ ) : (Σ _∪_ V) _∗_ _→_ Σ _∪_ V
and is recursively defined as:

TDec _pt_ ( _σ_ ; _θ_ ) =




TDec _p_ ( _σ ·_ TDec _pt−_ 1( _σ_ ; _θ_ ); _θ_ )

(if _t >_ 1)

TDec _p_ ( _σ_ ; _θ_ )

(if _t_ = 1)



where _·_ denotes token concatenation over Σ _∪_ V.

This definition highlights the iterative nature of autoregressive generation, Furthermore, by restricting the codomain
to the last token, this formulation aligns with the objectives
of this study, emphasizing the relationship between autoregressive behavior and formal language recognition. From
now on, when the context is clear, we simply write TDec.

**4.3. The Language Recognized by Transformer Decoder**

We now define the language recognized by a _p_ -precision
Transformer Decoder with a certain parameter _θ_ and _t_ -times
decode steps, based on the definition 4.1 and 4.2.

**Definition 4.3.** The language recognized by such a _t_ -times
autoregressive transformer decoder with a certain parameters _θ_ over _p_ -precision, TDec _p_ ( _·_ ; _θ_ ), _L_ (TDec _p_ ( _·_ ; _θ_ )) is
defined as:

_L_ (TDec _pt_ ( _·_ ; _θ_ ) _, F_ )

= _{w_ _∈_ Σ _[∗]_ _|∃r_ _≤_ _t_ ( _|w|_ ) _._ TDec _pr_ ( _w·⟨_ sep _⟩_ ; _θ_ ) _∈_ _F_ _}_ [(1)]

where _F_ _⊆_ V is the nonempty set of accept token. Typically,
_F_ may include tokens such as _⟨_ eos _⟩_ or other special markers
representing accept tokens.

Definition 4.3 states that an input string _w_ is accepted if the
output sequence TDec( _w · ⟨_ sep _⟩_ ) _∈_ (Σ _∪_ V) _[∗]_ contains at

least one accept token from _F_, within _t_ ( _|w|_ ) times or less
autoregression.

It is important to note that the special token _⟨_ sep _⟩_ is explicitly appended to the input sequence to distinguish the
decoding sequence. Additionally, the length of the output
sequence increase by a time function _t_ : N _→_ N, which
maps the input sequence _w_ to a maximum allowable number
of decoding steps. For example: If _t_ ( _n_ ) = _n_ [2], polynomially
many decoding steps are permitted. If _t_ ( _n_ ) = _c_, decoding is
restricted to a constant steps, regardless of the input length.

**Example 4.4.** _Let the time function be a constant function_
_t_ ( _n_ ) = 4 _,_ _and_ _the_ _set_ _of_ _accept_ _tokens_ _be_ _F_ = _{⟨_ eos _⟩}._
_Given that the output sequences of_ TDec _for the input se-_
_quences ”aabb” and ”aa” are as follows:_

TDec( _aabb⟨_ sep _⟩_ ) = _aba⟨_ eos _⟩_ _. . ._

TDec( _aa⟨_ sep _⟩_ ) = _aaaa . . ._

_In this case, the Transformer accepts only “aabb”._

**4.4. Confirmation of Constraints**

All other hyperparameters, such as the number of layers
_L_ _≥_ 2, the model dimension _d_, and attention heads, are
fixed as _O_ (1), regardless of the input sequence length _n_ . In
summary, this study incorporates certain modifications:

  - _Exclusion of positional encoding (NoPE; only for § 5)_

  - Two-layer Transformer Block, Single-head Attention
without pre-norm configuration (§ 4.1)

  - Causal masking for attention computation, and softmax
function within the Attention mechanism (§ 4.1)

  - Greedy Search decoding (Definition 4.3)

This formalization bridges the autoregressive generation
mechanism with the theoretical analysis of language recognition. In subsequent sections, we explore the expressivity
of Transformer Decoder models within this framework.

**5. Main Result 1:** **Finiteness of Fixed-Precision**
**Transformer without PE**

In this section, we present our first main result concerning the expressivity of Transformers under fixed-precision
arithmetic and softmax-based attention mechanisms. This
result establishes a direct correspondence between the class
of languages recognized by Transformers and finite or cofinite languages (Theorem 5.2) under a natural assumption
(Assumption 5.1).

**Infinity-Free Parameter Assumption** We begin by introducing a natural assumption regarding the parameters of the
attention layers in Transformers.

4

220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274

**Assumption** **5.1** (Infinity-Freeness) **.** For each attention
layer, the matrix product of query and key vectors is always greater than minus infinity ( _−_ inf _∈_ D _p_ ):

_d_
_∀y, y_ _[′]_ _∈_ D _p_ _._ _Q_ ( _y_ ) _K_ ( _y_ _[′]_ ) [T] = _±_ inf (2)

_d_
where _d_ _∈_ N is the model dimension, and _Q, K_ : D _p_ _→_
_d_
D _p_ are the query and key affine transformations.

This assumption depends only on the parameters of the
query and key affine transformations. It generally holds for
most trained Transformer models.

**Theorem 5.2** (Finiteness and Co-finiteness of Languages
Recognized by Transformer Decoder) **.** _Assume that Assump-_
_tion 5.1 holds._ _Under this assumption, the languages rec-_
_ognized_ _by_ _any_ _Transformer_ _decoders_ _is_ _exactly_ _finite_ _or_
_co-finite_ _languages._ _Specifically,_ _the_ _following_ _two_ _state-_
_ments hold:_

_1._ **(upper** **bound)** _For_ _any_ _p_ _∈_ N _, t_ ( _n_ ) _∈_ Ω(1) _, θ_ _∈_
Params _, F_ _⊆_ V _, there exists a finite or co-finite lan-_
_guage Lf_ _such that L_ (TDec _, F_ ) = _Lf_ _._

_2._ **(lower** **bound)** _For_ _any_ _finite_ _or_ _co-finite_ _language_
_L_ _[′]_ _f_ _[, there exist parameters][ p][′]_ _[∈]_ [N] _[, t][′]_ [(] _[n]_ [)] _[ ∈]_ [Ω(1)] _[, θ][′]_ _[∈]_
Params _, F_ _[′]_ _⊆_ V _such that L_ _[′]_ _f_ [=] _[ L]_ [(TDec] _[, F][ ′]_ [)] _[.]_

Theorem 5.2 represents a key result of this study. It states
that, under the infinity-freeness (Assumption 5.1) and with
fixed precision _p_, the class of languages recognized by Transformer decoders aligns exactly with the class of finite and
co-finite languages, regardless of the specific parameters,
the number of decoding steps, or the set of accept states.
Or vice versa, that means when the input length exceeds a
certain number, transformer model cannot distinguish the
inputs.

The two claims of Theorem 5.2 are proved in § 5.1 and § 5.2,
respectively.

**5.1. Proof of the Upper Bound under Assumption 5.1**

**Lemma 5.3.** _Suppose_ _**Assumption 5.1**_ _holds._ _Then there_
_exists an integer L ∈_ N _with the following property:_

_For_ any _two_ _inputs_ _w, w_ _[′]_ _∈_ Σ _[∗]_ _with_ _|w|, |w_ _[′]_ _|_ _≥_ _L,_ _the_
_Transformer_ _decoder_ TDec _pt_ [�] _w_ _· ⟨_ sep _⟩_ ; _θ_ - _produces_ _the_
same _output_ _tokens_ _as_ _it_ _does_ _for_ TDec _pt_ [�] _w_ _[′]_ _· ⟨_ sep _⟩_ ; _θ_ - _,_
_provided w and w_ _[′]_ _share the same final character._

_Proof._ Let us denote the final token of the input as _v_ _∈_
Σ _∪{⟨_ sep _⟩}_ . By Assumption 5.1, we know that for any vectors _y, y_ _[′]_ _∈_ (D _p_ ) _[d]_, the dot-product _Q_ ( _y_ ) _K_ ( _y_ _[′]_ ) _[⊤]_ = _−_ inf.
In particular, we can choose constants _α, β_ in (D _p_ ) _[d]_ (related to the embedding of _v_ ) such that the repeated sum

of exp( _Q_ ( _α_ ) _K_ ( _β_ ) _[⊤]_ ) over enough positions saturates the
_p_ -precision range to +inf. Hence we define _L_ to be the
minimum length at which this “+inf sum” occurs in the
causal masking scenario.

Let _w_ be any string with _|w|_ _≥_ _L_ . When the decoder
at time-step _r_ attends over all previously seen tokens
( _⟨_ sep _⟩_ appended at the end), the _softmax_ _denominator_ in
Attn( _qv, Kw, Vw_ ) accumulates

_|w|_ +1

   - exp� _qv Kj_ _[⊤]_    

_j_ =1

and by the definition of _L_, this sum diverges
to +inf in _p_ -precision. Consequently, the fraction exp� _qv K|_ _[⊤]_ _w|_ - _/_ (+inf) is effectively 0 in _p_ precision,making the final token’s contribution vanish.
Repeating this for each layer (and for each of the _t_ ( _|w|_ )
auto-regressive decoding steps) shows that any distinct
differences in _w_ vs. _w_ _[′]_ ( _provided_ their last character is the
same) are overshadowed as _|w| →∞_ .

Thus if _w, w_ _[′]_ _∈_ Σ _[∗]_ both have length _≥_ _L_ and share the same
last symbol _v_, the decoder outputs TDec _[t]_ ( _w · ⟨_ sep _⟩_ ; _θ_ ) and
TDec _[t]_ ( _w_ _[′]_ _·⟨_ sep _⟩_ ; _θ_ ) coincide. In other words, once the input
length is beyond _L_, the model cannot further distinguish
among long strings ending in the same symbol.

**Why Lemma 5.3 implies finite/co-finite recognition.** By
Lemma 5.3, all strings of length _≥_ _L_ that share a final
character are mapped to the _same_ sequence of output tokens
under the _t_ ( _|w|_ )-step decoding. Hence, if for some long
string _w_ the decoder _accepts_ (i.e. produces a token in _F_ _⊆_
V), then _all sufficiently long strings_ with the same last letter
are also accepted. Thus we obtain either:

  - A _co-finite_ pattern: the model rejects only finitely
many strings (those of length _<_ _L_, plus possibly
a few last-letter classes among the long strings), so
_L_ (TDec _[t]_ _p_ [(] _[·]_ [;] _[ θ]_ [)] _[, F]_ [)][ is co-finite.]

  - A _finite_ pattern: the model accepts only finitely many
cases (if it rejects all length _≥_ _L_ strings except perhaps
a handful).

In both cases, the recognized language is either finite or
co-finite.

**Remark** **on** _⟨_ sep _⟩_ **.** Including a special terminal token
_⟨_ sep _⟩_ in the input helps ensure that the “last symbol” alignment is explicit. Without it, one might rely on actual last
letters in Σ, and the argument becomes a suffix-based distinction rather than a crisp boundary. Our Definition 4.3
ensures that _w⟨_ sep _⟩_ standardizes the final token ( _or_ the last
letter in _w_ if no _⟨_ sep _⟩_ is appended),leading to a simpler
classification at large lengths.

5

_275_
_276_
_277_
_278_
_279_
_280_
_281_
_282_
_283_
_284_
_285_
_286_
_287_
_288_
_289_
_290_
_291_
_292_
_293_
_294_
_295_
_296_
_297_
_298_
_299_
_300_
_301_
_302_
_303_
_304_
_305_
_306_
_307_
_308_
_309_
_310_
_311_
_312_
_313_
_314_
_315_
_316_
_317_
_318_
_319_
_320_
_321_
_322_
_323_
_324_
_325_
_326_
_327_
_328_
_329_

**5.2. Proof of the Lower Bound under Assumption 5.1**

In this subsection, we show that a Transformer decoder
can recognize _any_ finite or co-finite language _Lf_ _⊆_ Σ _[∗]_ in
_constant_ (one or two) decoding steps. Formally, we will
construct a _p_ -precision Transformer decoder that outputs a
special “accept” token (e.g. _⟨_ eos _⟩_ ) if and only if the input
string belongs to _Lf_ .

**Lemma 5.4.** _Let Lfin_ _⊆_ Σ _[∗]_ _be any finite language._ _Then_
_there exist:_ _a precision parameter p ∈_ N _, a parameter set_
_θ_ _∈_ Params _, and a set of accept state tokens F_ _⊆_ V _, such_
_that the decoder recognizes Lfin_ _in exactly_ one _decoding_
_step._ _That is, L_ ( _TDec, F_ ) = _Lfin._

_Proof._ We design a two-layer Transformer decoder that
first ( _i_ ) accumulates sufficient information (e.g. a partial
sum or isomorphic encoding of the entire input string), and
then ( _ii_ ) employs a feed-forward network (FFN) to map
that information to a binary output: namely “ _w_ _∈_ _L_ ” or
“ _w_ _∈/_ _L_ ”. When _w_ _∈_ _Lfin_, the decoder emits a special
token ( _⟨_ eos _⟩_ ) on the single decoding step; otherwise it does
not.

**Embedding** **layer.** Suppose the input tokens are
_w_ 1 _, . . ., wn_ _∈_ Σ. Let _p_ be large enough to accommodate all
numerics (we will specify _p_ in a moment). For each token
_wi_, define its embedding vector as

_d_
**x** _i_ := [0 _,_ emb( _wi_ ) ] _∈_ D _p_

(3)
_d−_ 1
where emb( _wi_ ) _∈_ D _p_ _._

The extra leading coordinate (0) will be used to store positions or partial sums in subsequent layers.

**First** **attention** **layer.** We apply a _uniform_ _attention_ to
gather position-related or partial-sum information. For instance, let the query, key, and value transformations be:

_Q_ ( **x** ) = **1** _,_ _K_ ( **x** ) = **1** _,_

(4)
_d_
_V_ ( **x** ) = [ 1 _,_ 0 _, . . .,_ 0 ] _∈_ D _p_

for all input vectors **x** . Then, under causal masking (each
**x** _i_ only attends to **x** 1: _i_ ), the attention output for **x** _i_ is:

Attn� _Q_ ( **x** _i_ ) _, K_ ( **x** 1: _i_ ) _, V_ ( **x** 1: _i_ )� =  - 1 _i_ _[,]_ [0] _[, . . .,]_ [ 0]  - _._ (5)

Thus, after adding the residual connection, the layer output
becomes:

**a** [1] _i_ [=]    - 1 _i_ _[,]_ [0] _[, . . .,]_ [ 0]    - + **x** _i_ =    - 1 _i_ _[,]_ [emb(] _[w][i]_ [)]    - _._ (6)

**First feed-forward network.** We design an FFN so that

**z** [1] _i_ [= FFN]      - **a** [1] _i_      - =      - 0 _,_ emb( _i_ _wi_ )      - (7)

This step ensures each position’s embedding is scaled by
1 _/i_ and placed in the tail part of the vector.

**Second attention layer.** We next use the _n_ -th token **x** _n_ (or
similarly the “final step”) to attend over all **z** [1] 1 _[, . . .,]_ **[ z]** _n_ [1] [.] [Let]
_Q_ ( **x** ) = **1** _,_ _K_ ( **x** ) = **1** _,_ _V_ ( **x** ) = **x** . Hence,

**a** [2] _n_ [= Attn]    - _Q_ ( **z** [1] _n_ [)] _[, K]_ [(] **[z]** [1] 1: _n_ [)] _[, V]_ [ (] **[z]** [1] 1: _n_ [)]    

= _n_ [1] - 0 _,_

_n_

_k_ =1

emb( _wk_ ) - (8)

_k_

Since we choose _p_ large enough, _n_ [1] [=] _[p]_ [0][ in the] _[ p]_ [-precision]

sense.

**Remark.** The partial sum [�] _k_ _[n]_ =1 emb( _kwk_ ) can be seen as

carrying isomorphic information about ( _w_ 1 _, . . ., wn_ ), assuming a suitable injection or universal approximation property (we treat details abstractly here).

**Second feed-forward network.** Finally, we use a universalapproximation argument: there is an MLP or FFN that can
decode **a** [2] _n_ _[∼]_ _[w]_ [1:] _[n]_ [and output 1 iff] _[ w]_ _[∈]_ _[L][fin]_ [:]

FFN [2][�] **a** [2] _n_ - =

1 (if _w_ _∈_ _Lfin_ ) _,_
(9)
0 (otherwise) _._

We then interpret output “1” as a special accept token (e.g.
_⟨_ eos _⟩_ ) in the output layer. Hence, the entire decoder recognizes exactly _Lfin_

**Extension** **to** **co-finite** **languages.** For a co-finite language _Lcofin_, we simply invert the behavior: almost all
strings map to “1” (accept), while the finite exceptional set
Σ _[∗]_ _\ Lcofin_ maps to “0.” A parallel argument with slight
modifications (where the second FFN outputs 1 for nearly
all inputs, except a finite listed set of strings) completes the
proof.

**Conclusion.** By combining these constructions, we see
that any finite or co-finite language _Lf_ _⊆_ Σ _[∗]_ can indeed be
recognized by a _p_ -precision, two-layer Transformer decoder
_in one or two decoding steps_ . Thus, for such _Lf_, we have
_Lf_ = _L_ (TDec) for some parameter choice and constant
decode budget.

In summary, _Assumption_ _5.1_ plus the no-positionalencoding policy forces the Transformer decoder to unify all
sufficiently long strings with identical trailing tokens. Hence
the language recognized cannot exceed finite or co-finite
sets, completing the proof of the upper bound.

6

330
331
332
333
334
335
336
337
338
339
340
341
342
343
344
345
346
347
348
349
350
351
352
353
354
355
356
357
358
359
360
361
362
363
364
365
366
367
368
369
370
371
372
373
374
375
376
377
378
379
380
381
382
383
384

**6. Main Result 2:** **The language recognized by**
**fixed-precision decoder**

**6.1. Lower Bound for asm. 5.1 and APE**

We now prove that a Transformer decoder _with_ _Assump-_
_tion 5.1 and some APE_ can recognize any _cyclic language_ .

**Theorem 6.1.** _For any m-cyclic language Lc, there exist_
_some Transformer with asm. 5.1 and Abusolute Positional_
_Encoding,_ TDec _[′]_ _such that Lc_ = _L_ (TDec _[′]_ _,_ Fc) _for some_
_set of special tokens Fc_ _⊆_ V

_Proof Sketch._ **APE** **to** **distinguish** **positions** **mod** _**m**_ For
given _m_ -cyclic language, prepare suitable APE such that
have periodicity (e.g., sinusoidal embeddings (Vaswani
et al., 2017; Chiang et al., 2023)), and the Transformer
can effectively identify each position’s residue class modulo _m_ Hence, for index _i_ and _j_, if _i_ _≡_ _j_ (mod _m_ ), their
positional encodings can be made same so that the network
recognizes the same residue class.

**Attention mechanism** In other words, for the head corresponding to residue _r_, only tokens _xi_ with _i ≡_ _r_ (mod _m_ )
receive a high attention score. Under the “inf-free” condition, no key–query product becomes _−_ inf, so we can rely on
softmax-based attention to highlight precisely those tokens
that belong to the right residue class.

**FFN** **to** **implement** **the** _m_ **-cyclic** **condition** Since an _m_ cyclic language determines acceptance based on how symbols appear in these residue classes, the final feed-forward
network can be crafted to check the patterns aggregated from
each class. Concretely, if _Lc_ says “Positions _≡_ _r_ (mod _m_ )
must contain letter _a_ ” or “must exclude letter _b_,” then after the multi-head attention, the hidden representation has
sufficient information to confirm or deny these constraints.

Putting it all together, the Transformer obtains positionresidue awareness from APE, employs attention to gather
all tokens of each residue class, and checks with a FFN
whether the cyclic criteria are satisfied. Thus, for any _m_ cyclic language _Lc_, we construct a suitable Transformer
decoder (satisfying inf-free and using APE) so that _Lc_ is
recognized exactly by that model, completing the proof.

**6.2. Lower Bound for NoPE: Letter-set Languages**

We now prove that a Transformer decoder _without any posi-_
_tional encoding_ (NoPE) _and Assumption 5.1_ can recognize
any _letter-set_ _language_ . Formally, a _letter-set_ _language_
_LS_ _⊆_ Σ _[∗]_ is one where acceptance only depends on which
_letters_ (symbols) appear in _w_ (not on their order or count).

**Theorem** **6.2.** _For_ _any_ _letter-set_ _language_ _LS,_ _there_ _ex-_
_ist_ _some_ _Transformer_ _decoders_ _with_ _No_ _Positional_ _En-_
_coding_ _and_ _without_ _Assumption_ _5.1,_ TDec _[′′]_ _such_ _that_
_Ls_ = _L_ (TDec _[′′]_ _,_ Fs) _for some set of special tokens Fs_ _⊆_ V

_Proof Sketch._ A letter-set language is determined solely by
the set of unique letters in the input string. In a Transformer
decoder without positional encoding, identical letters are
mapped to identical embedding vectors, irrespective of their
positions in the input sequence. Consequently, the model
cannot distinguish whether _a_ appears as the first or fifth
letter, but it can identify whether _a_ is present in the input
at all. By processing embeddings, the model can determine
the existence of each letter without tracking its count or
position.

Using attention and feed-forward layers, the model can consolidate these embeddings to produce a “presence flag” for
each letter. The flag is set based on whether the embedding is zero or non-zero. Thus, we employ the non-finite
floating-point value inf in the denominator during the transition computation to make the flags zero aligning with the
discussion in Lamma 5.3. This is why the Assumption 5.1
is removed. For example, if _a_ appears anywhere in the
sequence, a specific hidden vector state can be activated to
indicate its presence.

Since letter-set languages are defined by finite logical combinations of conditions on letter presence, the final feedforward and output layers can evaluate these conditions. For
example, the model can output an accept token if the presence flags match the required subset _S_, or reject otherwise.
This process effectively ignores order and frequency, focusing solely on whether each required letter is present at least
once.

The lack of positional encoding aligns naturally with the
requirements of letter-set languages. A NoPE Transformer
focuses on whether a given letter appears, without being
influenced by order or frequency. Even if a letter _a_ appears
multiple times, the model only needs a single bit of information (” _a_ exists”) to make its decision. By aggregating these
presence flags, the Transformer can determine whether the
input satisfies the rules of the letter-set language.

Thus, a NoPE Transformer can recognize any letter-set language, using its ability to abstract away positional information and focus on the presence of letters.

**7. Discussions**

**7.1. What is the Key Module in Transformers?**

Although numerous studies have advanced our understanding of Transformers, a fundamental question remains:
_“Which_ _architectural_ _component_ _primarily_ _contributes_ _to_
_their expressivity?”_ Despite extensive research on elements
like attention mechanisms, layer normalization, and embedding schemes, there is no universal consensus on _what_
_exactly_ determines a language model’s ability to capture
complex linguistic phenomena.

7

385
386
387
388
389
390
391
392
393
394
395
396
397
398
399
400
401
402
403
404
405
406
407
408
409
410
411
412
413
414
415
416
417
418
419
420
421
422
423
424
425
426
427
428
429
430
431
432
433
434
435
436
437
438
439

Bhattamishra et al. (2020b) focused on the Turingcompleteness and the necessity of various architectural components and highlighted the crucial role of residual connections in maintaining expressivity. They also demonstrated
that Transformers without explicit positional encoding but
with positional masking remain Turing-complete. Similarly, Chiang et al. (2023) highlighted the importance of
numerical precision (fixed vs arbitrary) and showed that the
expressivity of such an encoder Transformer can be tightly
upper-bounded by the language class FOC[+; MOD], a firstorder logic with counting quantifiers, addition, and modular
arithmetic.

A crucial difference emerges when comparing their results
to ours: they included _positional encoding_ (specifically a
sinusoidal scheme), which allowed the model to handle periodic information effectively. In this study, We adopted a
constant precision scheme similar to Chiang et al. (2023).
Moreover, we introduced a reasonable practical Assumption 5.1. Building on these settings,we identified a Transformer setup capable of recognizing the minimal language,
namely finite or co-finite languages, without any positional
encodings (§ 5). This setup closely resembles real-world
Transformers, leading us to hypothesize that practical Transformers may inherently be restricted to recognizing finite
languages, functioning as highly efficient lookup tables.
While prior studies (Bhattamishra et al., 2020b; Kazemnejad et al., 2024)) demonstrated the practical effectiveness
of NoPE, our theoretical analysis suggests that NoPE has
inherent limitations in enhancing expressivity.

Next, we examined how adding absolute positional encoding
(APE) and removing the assumption affected the tendency to
restrict recognition to finite languages (§ 6). However, even
with these additions, expressivity increased only slightly, as
fixed-precision still constrains expressivity to near-finiteness.
Our findings show that restricting precision from logarithmic (Merrill & Sabharwal, 2024b) to constant results in a
significant loss of expressiveness. Furthermore, this loss
increases as the number of decoding iterations grows, noting
that expressivity reaches P when _t ∈O_ ( _n_ _[c]_ ).

**7.2. Languge Modeling**

Throughout this work, we frame the Transformer as a _lan-_
_guage recognizer_, addressing the membership problem in a
more formal sense rather than as a _language generator_ .

In practice, particularly in language modeling, a decoderbased Transformer typically produces tokens probabilistically, generating text rather than deciding membership in
a formal language. In fact, research on the expressivity of
language modeling exists (Svete & Cotterell, 2024; Nowak
et al., 2024). While our “recognizer” viewpoint diverges
somewhat from typical usage, bridging these two outlooks
more rigorously remains a key objective for future research.

**7.3. Potential Extensions**

We acknowledge that our current setup is simplified, focusing on a limited subset of Transformer components: attention masking, the absence of layer normalization, and
no extensive multi-head or multi-layer structure. In realworld architectures, additional architectural features could
significantly impact expressivity.

Furthermore, we have identified gaps (Table 1). A natural
extension involves clarifying how these additional mechanisms, such as relative positional encoding or the softmax-tohardmax transition, might shift the upper and lower bounds
on expressivity. We believe our fundamental approach can
be adapted to investigate such enhancements, while leaving
precise formalization and empirical validation for future
work.

**8. Conclusion**

In this work, we examined the expressivity of fixedprecision Transformers to investigate their practical implications. To achieve this, we introduced three constraints:
fixed-precision floating-point arithmetic, a reasonable assumption 5.1 regarding query-key parameters, and the presence or absence of positional encoding.

In § 5, we demonstrated that Transformers operating under
the constraints (Fixed-precision + Assumption 5.1 + NoPE)
can recognize only finite or co-finite languages. In § 6, we
further proved the role of Assumption 5.1 and also positional
encoding, as relaxing either of these constraints slightly
enhances expressivity.

These findings suggest that these constraints impose fundamental limitations on Transformer expressivity. Future
research could extend this analysis to language modeling or
investigate how alternative modules and hardmax replacements influence expressivity.

**References**

Ieee standard for floating-point arithmetic. _IEEE Std 754-_
_2019 (Revision of IEEE 754-2008)_, pp. 1–84, 2019. doi:
10.1109/IEEESTD.2019.8766229.

Ackerman, J. and Cybenko, G. A survey of neural networks
and formal languages, 2020. [URL https://arxiv.](https://arxiv.org/abs/2006.01338)
[org/abs/2006.01338.](https://arxiv.org/abs/2006.01338)

Bhattamishra, S., Ahuja, K., and Goyal, N. On the Ability
and Limitations of Transformers to Recognize Formal
Languages. In Webber, B., Cohn, T., He, Y., and Liu, Y.
(eds.), _Proceedings_ _of_ _the_ _2020_ _Conference_ _on_ _Empiri-_
_cal Methods in Natural Language Processing (EMNLP)_,
pp. 7096–7116, Online, November 2020a. Association
for Computational Linguistics. doi: 10.18653/v1/2020.

8

440
441
442
443
444
445
446
447
448
449
450
451
452
453
454
455
456
457
458
459
460
461
462
463
464
465
466
467
468
469
470
471
472
473
474
475
476
477
478
479
480
481
482
483
484
485
486
487
488
489
490
491
492
493
494

emnlp-main.576. URL [https://aclanthology.](https://aclanthology.org/2020.emnlp-main.576/)
[org/2020.emnlp-main.576/.](https://aclanthology.org/2020.emnlp-main.576/)

Bhattamishra, S., Patel, A., and Goyal, N. On the
computational power of transformers and its implications in sequence modeling. In Fernandez,´ R. and
Linzen, T. (eds.), _Proceedings_ _of_ _the_ _24th_ _Conference_
_on Computational Natural Language Learning_, pp. 455–
475, Online, November 2020b. Association for Computational Linguistics. doi: 10.18653/v1/2020.conll-1.
37. URL [https://aclanthology.org/2020.](https://aclanthology.org/2020.conll-1.37/)
[conll-1.37/.](https://aclanthology.org/2020.conll-1.37/)

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan,
J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G.,
Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu,
J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M.,
Gray, S., Chess, B., Clark, J., Berner, C., McCandlish,
S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In _Proceedings of_
_the_ _34th_ _International_ _Conference_ _on_ _Neural_ _Informa-_
_tion Processing Systems_, NIPS ’20, Red Hook, NY, USA,
2020. Curran Associates Inc. ISBN 9781713829546.

Chiang, D. and Cholak, P. Overcoming a theoretical limitation of self-attention. In Muresan, S., Nakov, P., and
Villavicencio, A. (eds.), _Proceedings of the 60th Annual_
_Meeting of the Association for Computational Linguistics_
_(Volume_ _1:_ _Long_ _Papers)_, pp. 7654–7664, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.527. [URL https:](https://aclanthology.org/2022.acl-long.527/)
[//aclanthology.org/2022.acl-long.527/.](https://aclanthology.org/2022.acl-long.527/)

Chiang, D., Cholak, P., and Pillay, A. Tighter bounds on the
expressivity of transformer encoders. In _Proceedings of_
_the 40th International Conference on Machine Learning_,
ICML’23. JMLR.org, 2023.

Chiang, D., Rawski, J., Strobl, L., and Yang, A.
Esslli 2024, 2024. [https://sleynas.com/](https://sleynas.com/esslli-2024-summer-school-course)
[esslli-2024-summer-school-course](https://sleynas.com/esslli-2024-summer-school-course) (202501 viewed).

Cybenko, G. Approximation by superpositions of a sigmoidal function. _Mathematics of Control, Signals, and_
_Systems (MCSS)_, 2(4):303–314, December 1989. ISSN
0932-4194. doi: 10.1007/BF02551274. URL [http:](http://dx.doi.org/10.1007/BF02551274)
[//dx.doi.org/10.1007/BF02551274.](http://dx.doi.org/10.1007/BF02551274)

Feng, G., Zhang, B., Gu, Y., Ye, H., He, D., and Wang, L.
Towards revealing the mystery behind chain of thought: a
theoretical perspective. In _Proceedings of the 37th Inter-_
_national Conference on Neural Information Processing_
_Systems_, NIPS ’23, Red Hook, NY, USA, 2023. Curran
Associates Inc.

Hahn, M. Theoretical limitations of self-attention in neural
sequence models. _Transactions_ _of_ _the_ _Association_ _for_
_Computational Linguistics_, 8:156–171, 01 2020. ISSN
2307-387X. doi: 10.1162/tacl ~~a~~ ~~0~~ 0306. [URL https:](https://doi.org/10.1162/tacl_a_00306)
[//doi.org/10.1162/tacl_a_00306.](https://doi.org/10.1162/tacl_a_00306)

Hao, Y., Angluin, D., and Frank, R. Formal language
recognition by hard attention transformers: Perspectives
from circuit complexity. _Transactions_ _of_ _the_ _Associa-_
_tion for Computational Linguistics_, 10:800–810, 07 2022.
ISSN 2307-387X. doi: 10.1162/tacl ~~a~~ ~~0~~ 0490. URL
[https://doi.org/10.1162/tacl_a_00490.](https://doi.org/10.1162/tacl_a_00490)

Hornik, K., Stinchcombe, M., and White, H. Multilayer
feedforward networks are universal approximators.
_Neural Networks_, 2(5):359–366, 1989. ISSN 0893-6080.
doi: https://doi.org/10.1016/0893-6080(89)90020-8.
URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/0893608089900208)
[science/article/pii/0893608089900208.](https://www.sciencedirect.com/science/article/pii/0893608089900208)

Kajitsuka, T. and Sato, I. Are transformers with one layer
self-attention using low-rank weight matrices universal
approximators?, 2024. [URL https://arxiv.org/](https://arxiv.org/abs/2307.14023)
[abs/2307.14023.](https://arxiv.org/abs/2307.14023)

Kazemnejad, A., Padhi, I., Ramamurthy, K. N., Das, P.,
and Reddy, S. The impact of positional encoding on
length generalization in transformers. In _Proceedings of_
_the 37th International Conference on Neural Information_
_Processing_ _Systems_, NIPS ’23, Red Hook, NY, USA,
2024. Curran Associates Inc.

Merrill, W. Formal language theory meets modern nlp, 2021.
[URL https://arxiv.org/abs/2102.10094.](https://arxiv.org/abs/2102.10094)

Merrill, W. Formal languages and the nlp black box.
In _Developments_ _in_ _Language_ _Theory:_ _27th_ _Interna-_
_tional_ _Conference,_ _DLT_ _2023,_ _Umea,˚_ _Sweden,_ _June_
_12–16, 2023, Proceedings_, pp. 1–8, Berlin, Heidelberg,
2023. Springer-Verlag. ISBN 978-3-031-33263-0. doi:
10.1007/978-3-031-33264-7 ~~1~~ . URL [https://doi.](https://doi.org/10.1007/978-3-031-33264-7_1)
[org/10.1007/978-3-031-33264-7_1.](https://doi.org/10.1007/978-3-031-33264-7_1)

Merrill, W. and Sabharwal, A. The parallelism tradeoff:
Limitations of log-precision transformers. _Transactions_
_of the Association for Computational Linguistics_, 11:531–
545, 2023. doi: 10.1162/tacl ~~a~~ ~~0~~ 0562. URL [https:](https://aclanthology.org/2023.tacl-1.31/)
[//aclanthology.org/2023.tacl-1.31/.](https://aclanthology.org/2023.tacl-1.31/)

Merrill, W. and Sabharwal, A. A logic for expressing logprecision transformers. In _Proceedings of the 37th Inter-_
_national Conference on Neural Information Processing_
_Systems_, NIPS ’23, Red Hook, NY, USA, 2024a. Curran
Associates Inc.

Merrill, W. and Sabharwal, A. The expressive power of
transformers with chain of thought, 2024b. [URL https:](https://arxiv.org/abs/2310.07923)
[//arxiv.org/abs/2310.07923.](https://arxiv.org/abs/2310.07923)

9

495
496
497
498
499
500
501
502
503
504
505
506
507
508
509
510
511
512
513
514
515
516
517
518
519
520
521
522
523
524
525
526
527
528
529
530
531
532
533
534
535
536
537
538
539
540
541
542
543
544
545
546
547
548
549

Merrill, W., Sabharwal, A., and Smith, N. A. Saturated
transformers are constant-depth threshold circuits. _Trans-_
_actions_ _of_ _the_ _Association_ _for_ _Computational_ _Linguis-_
_tics_, 10:843–856, 08 2022. ISSN 2307-387X. doi:
10.1162/tacl ~~a~~ 00493. [URL https://doi.org/10.](https://doi.org/10.1162/tacl_a_00493)
[1162/tacl_a_00493.](https://doi.org/10.1162/tacl_a_00493)

Nowak, F., Svete, A., Butoi, A., and Cotterell, R. On
the representational capacity of neural language models
with chain-of-thought reasoning. In Ku, L.-W., Martins,
A., and Srikumar, V. (eds.), _Proceedings_ _of_ _the_ _62nd_
_Annual_ _Meeting_ _of_ _the_ _Association_ _for_ _Computational_
_Linguistics (Volume 1:_ _Long Papers)_, pp. 12510–12548,
Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.
676. URL [https://aclanthology.org/2024.](https://aclanthology.org/2024.acl-long.676/)
[acl-long.676/.](https://aclanthology.org/2024.acl-long.676/)

Park, S., Yun, C., Lee, J., and Shin, J. Minimum width
for universal approximation, 2020. URL [https://](https://arxiv.org/abs/2006.08859)
[arxiv.org/abs/2006.08859.](https://arxiv.org/abs/2006.08859)

Press, O., Smith, N., and Lewis, M. Train short, test long:
Attention with linear biases enables input length extrapolation. In _International Conference on Learning Represen-_
_tations_, 2022. URL [https://openreview.net/](https://openreview.net/forum?id=R8sQPpGCv0)
[forum?id=R8sQPpGCv0.](https://openreview.net/forum?id=R8sQPpGCv0)

Perez, J., Barcel´ o, P., and Marinkovic, J.´ Attention is turingcomplete. _Journal of Machine Learning Research_, 22(75):
1–35, 2021. URL [http://jmlr.org/papers/](http://jmlr.org/papers/v22/20-302.html)
[v22/20-302.html.](http://jmlr.org/papers/v22/20-302.html)

Radford, A., Narasimhan, K., Salimans, T., and Sutskever,
I. Improving language understanding by generative pretraining. 2018.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang,
S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified
text-to-text transformer. _Journal_ _of_ _Machine_ _Learning_
_Research_, 21(140):1–67, 2020. URL [http://jmlr.](http://jmlr.org/papers/v21/20-074.html)
[org/papers/v21/20-074.html.](http://jmlr.org/papers/v21/20-074.html)

Strobl, L., Merrill, W., Weiss, G., Chiang, D., and Angluin, D. What formal languages can transformers express? a survey. _Transactions_ _of_ _the_ _Association_ _for_
_Computational Linguistics_, 12:543–561, 2024. doi: 10.
1162/tacl ~~a~~ ~~0~~ 0663. [URL https://aclanthology.](https://aclanthology.org/2024.tacl-1.30/)
[org/2024.tacl-1.30/.](https://aclanthology.org/2024.tacl-1.30/)

Svete, A. and Cotterell, R. Transformers can represent
_n_ -gram language models. In Duh, K., Gomez, H., and
Bethard, S. (eds.), _Proceedings of the 2024 Conference_
_of_ _the_ _North_ _American_ _Chapter_ _of_ _the_ _Association_ _for_

10

_Computational Linguistics:_ _Human Language Technolo-_
_gies_ _(Volume_ _1:_ _Long_ _Papers)_, pp. 6845–6881, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.
381. URL [https://aclanthology.org/2024.](https://aclanthology.org/2024.naacl-long.381/)
[naacl-long.381/.](https://aclanthology.org/2024.naacl-long.381/)

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I.
Attention is all you need. In Guyon, I., Luxburg, U. V.,
Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S.,
and Garnett, R. (eds.), _Advances in Neural Information_
_Processing Systems_, volume 30. Curran Associates, Inc.,
2017. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
[cc/paper_files/paper/2017/file/](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

[3f5ee243547dee91fbd053c1c4a845aa-Paper.](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
[pdf.](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

Wei, C., Chen, Y., and Ma, T. Statistically meaningful
approximation: a case study on approximating turing machines with transformers. In _Proceedings of the 36th Inter-_
_national Conference on Neural Information Processing_
_Systems_, NIPS ’22, Red Hook, NY, USA, 2022. Curran
Associates Inc. ISBN 9781713871088.

Yang, A. and Chiang, D. Counting like transformers: Compiling temporal counting logic into softmax transformers, 2024. [URL https://arxiv.org/abs/2404.](https://arxiv.org/abs/2404.04393)
[04393.](https://arxiv.org/abs/2404.04393)

Yang, A., Chiang, D., and Angluin, D. Masked hardattention transformers recognize exactly the star-free
languages, 2024. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2310.13897)
[2310.13897.](https://arxiv.org/abs/2310.13897)

Yao, S., Peng, B., Papadimitriou, C., and Narasimhan, K.
Self-attention networks can process bounded hierarchical
languages. In Zong, C., Xia, F., Li, W., and Navigli, R.
(eds.), _Proceedings_ _of_ _the_ _59th_ _Annual_ _Meeting_ _of_ _the_
_Association for Computational Linguistics and the 11th_
_International Joint Conference on Natural Language Pro-_
_cessing (Volume 1: Long Papers)_, pp. 3770–3785, Online,
August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.292. [URL https:](https://aclanthology.org/2021.acl-long.292/)
[//aclanthology.org/2021.acl-long.292/.](https://aclanthology.org/2021.acl-long.292/)

Yun, C., Bhojanapalli, S., Rawat, A. S., Reddi, S. J., and
Kumar, S. Are transformers universal approximators of
sequence-to-sequence functions?, 2020. [URL https:](https://arxiv.org/abs/1912.10077)
[//arxiv.org/abs/1912.10077.](https://arxiv.org/abs/1912.10077)

