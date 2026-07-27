# The Expressivity Of Fixed-Precision Transformers Without Positional Encoding

## Anonymous Authors1 Abstract

The primary objective of this study is to examine how practical constraints impact the expressivity of Transformers and to investigate their expressivity in real-world implementations. To achieve this, we analyze the expressivity of Transformer decoders operating under fixedprecision float arithmetic, an assumption regarding query-key parameters, and the presence or absence of positional encoding. Our findings reveal that, under fixed-precision and these constraints, Transformers are limited to recognizing finite or co-finite languages, a proper subclass of regular languages. While incorporating positional encoding or relaxing certain assumptions marginally enhances expressivity, the fundamental limitations imposed by fixed precision remain significant. These results underscore the gap between theoretical models and real-world implementations, suggesting that practical Transformers may be fundamentally constrained to recognizing only finite and co-finite languages, effectively functioning as little more than efficient lookup tables.

## 1. Introduction

The expressivity of Transformer models (Vaswani et al., 2017) has been further elucidated through recent theoretical analyses by comparing to the range of recognizable formal languages and solvable complexity classes. A series of studies has established upper and lower bounds on their expressivity under following settings. Perez et al. ´ (2021) is the first study to explore the expressivity of Transformers, proving their Turing-completeness using rational numbers, assuming infinite precision float. Subsequent studies adopting finite precision have provided more practical insights. For instance, Merrill & Sabharwal 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

Table 1. The upper and lower bound of the expressivity of fixedprecision Transformers. Chiang et al. (2023)
†identified the upper bound of normal Transformer encoder model (i.e. fixed-precision, sinusoidal positinoal encoding). In this study we showed that **bold** parts. "?" means the bound is not known.

Asm. Assumption. 5.1 - PE NoPE APE NoPE APE

| Asm.     | Assumption. 5.1   | -                   |      |            |
|----------|-------------------|---------------------|------|------------|
| PE       | NoPE              | APE                 | NoPE | APE        |
| FinCofin | FOC               |                     |      |            |
| Uppoer   | (§ 5.1)           | ?                   | ?    | [+; MOD] † |
| bound    | FinCofin          |                     |      |            |
| FinCofin |                   |                     |      |            |
| Lower    | m-cyclic          |                     |      |            |
| (§ 5.2)  |                   |                     |      |            |
| bound    | (§ 6.1)           | FinCofin letter-set | ?    |            |
| (§ 6.2)  |                   |                     |      |            |

(2023; 2024a) investigated logarithmic precision, which is finite but scales with input length n, and revealed that such Transformers are limited to much smaller circuit complexity classes, such as TC0or logical class FO(M), compared to Turing machines. Similarly, Chiang et al. (2023) examined fixed-precision Transformers and demonstrated that their tighter upper bounds are linked to logic FOC[+; MOD], which is an extension of first-order logic. Despite these theoretical advances, many studies rely on idealized conditions. This paper bridges the gap between these settings and real-world implementations, which impose significant constraints on processing and retaining information.

We investigate how the expressivity of Transformer decoders is shaped by the following practical constraints: fixed-precision floating-point numbers, positional encoding variations (APE, NoPE), and assumptions on parameter configurations(asm. 5.1). Our results indicate that expressivity depends on these constraints as follows (Table 1).

- Fixed-precision (e.g., fp32, bf16) limits recognition to finite and co-finite languages. {*a, b, ba, aab*} (§ 5)
- Absolute positional encoding extends recognition beyond finite and co-finite languages to cyclic languages. {*ab, abab, ababab, . . .* } (§ 6.1)
- Non-finite values (±inf) expand expressivity to letterset languages, capturing specific letter inclusion. {*abbb, ccac, bbac, . . .* } (§ 6.2)
Our findings extend prior results by highlighting the theoretical and practical implications of fixed-precision.

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 1

## 2. Related Work 2.1. Transformer Models And Expressivity

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 The computational capabilities of neural networks, covering RNNs, CNNs, and Transformers, have been extensively studied. The comprehensive surveys by Ackerman & Cybenko (2020); Merrill (2021; 2023) provide an in-depth overview of the expressivity of neural networks as a whole. Learnability is inherently bounded by expressivity, as the language that a model can recognize defines the boundaries of what it can effectively learn. Therefore, expressivity is not only a theoretical concern, but it is also of practical importance in guiding model design. A survey paper (Strobl et al., 2024) and lecture notes (Chiang et al., 2024) provide a comprehensive overview of recent advances in the study of Transformer expressivity, highlighting that expressivity is often analyzed in relation to three key areas: formal languages, circuit complexity, and logic. Formal languages Hahn (2020); Bhattamishra et al. (2020a;b); Yao et al. (2021); Chiang & Cholak (2022) primarily investigated the relationship between variants of hard- Transformers and formal languages such as PARITY and Dyck languages, which are commonly used benchmarks for expressivity. Feng et al. (2023); Merrill & Sabharwal (2024b) focused on the decoding time, inspired by chainof-thought reasoning (Wei et al., 2022), demonstrating that expressivity expands significantly with multiple decoding steps. Of particular interest, Nowak et al. (2024) examined how Transformers assign probabilities to strings in language modeling, identifying connections to probabilistic deterministic finite automata and probabilistic Turing machines.

Circuit complexity Another perspective comes from circuit complexity theory, which classifies computational problems based on their implementability within Boolean circuits of bounded depth and size. Hao et al. (2022) analyzed hard-Transformer variants, linking them to the tiny circuit class AC0. Merrill et al. (2022); Merrill & Sabharwal (2023)
extended this to more practical settings, showing that the saturated attention and logarithmic precision Transformers remain within TC0. Merrill & Sabharwal (2023) further suggested a fundamental parallelism trade-off, arguing that highly parallel architectures like Transformers may inherently face computational limits. Logic Chiang et al. (2023); Merrill & Sabharwal (2024a); Yang et al. (2024); Yang & Chiang (2024) have explored connections between Transformer models and first-order logic. These studies encode strings into Boolean variables and represent languages using logical frameworks such as first-order logic with counting quantifiers (FOC[+; MOD]).

While significant progress has been made, many studies rely on unrealistic assumptions such as infinite precision or hard-attention, leaving questions about their practical relevance.

## 2.2. Neural Networks And Function Approximation

A fundamental result in neural network theory is the universal approximation theorem, which states that any continuous function can be approximated arbitrarily well. While not the focus of our study, it provides essential context for understanding the broader capabilities of neural networks.

Feedforward networks Feedforward neural networks
(FFNs) play a central role in this context. Cybenko (1989);
Hornik et al. (1989) proved that FFNs with a single hidden layer and arbitrary nonlinear activations can universally approximate any Borel measurable or continuous function, given sufficient hidden units. Park et al. (2020) further identified the minimum width required for universal approximation, given the input and output dimensions. Transformers Recent work has extended universal approximation results to Transformers, with Yun et al. (2020) establishing their ability to approximate continuous sequence-to-sequence functions on compact domains and highlighting the crucial role of positional encoding in encoding order and circumventing permutation equivariance constraints. Kajitsuka & Sato (2024) later showed that even single-layer Transformers with low-rank weights can achieve such approximation power. Furthermore, Wei et al.

(2022) introduced the statistically meaningful approximation framework, addressing limitations in classical approximation theory by incorporating learnability constraints.

## 3. Preliminaries

In this section, we present the foundational concepts that support our theoretical results. For strings *w, w*′ ∈ Σ
∗ over the alphabet Σ, |w| denotes the length of the string, and ww′ denotes the concatenation. Furthermore, wt denotes the t-th character, and wi:j (*i, j* ∈ N) denotes the subsequence of w from the i-th to the j-th character.

## 3.1. Finite And Co-Finite Languages, Cyclic Language, Letter-Set Language

This subsection introduces *finite languages* and their dual, co-finite languages, along with *letter-set languages* and cyclic languages. These languages will play a central role in analyzing the expressivity of Transformers (§ 5, § 6).

Definition 3.1 (Finite Language). Let Σ be a finite alphabet. A language L ⊆ Σ
∗is called a *finite language* if and only if there exists k ∈ N such that for all stings w ∈ L, |w| ≤ k.

Definition 3.2 (Co-finite Language). Let Σ be a finite alphabet. A language L ⊆ Σ
∗is called a *co-finite language* if and only if its complement Σ
∗ \ L is a finite language.

Definition 3.3 (m-cyclic Language). Let Σ be a finite alphabet. A language L ⊆ Σ
∗is called a m-*cyclic language* if and only if for some m ∈ N, for all *w, w*′ ∈ L and for all 0 ≤ i ≤ max(|w|, |w
′|), wi ≡ w
′i mod m holds.

Definition 3.4 (Letter-set Language). Let Σ be a finite alphabet. A language L ⊆ Σ
∗is called a *letter-set language* if and only if for some set of letters A ⊆ Σ, for all w ∈ L includes all of the letters in A.

Example 3.5. The following languages L, L′ *over* Σ =
{a, b} *are co-finite languages:*
L = Σ∗\ {a, b, ab, aab}
L
′ = {w ∈ Σ
∗| |w| ≥ 3}
Similarly, the following language L{a,b} *over* Σ = {*a, b, c*} is letter-set language and L3 is 3*-cyclic language:*
L3 = (abc)
∗
L{a,b} = {w ∈ Σ
∗| w has both a and b}

## 3.2. P**-Precision Float-Point Numbers**

Now we define the rigorous mathematical framework for representing and manipulating numerical values under finite precision constraints, following (Merrill & Sabharwal, 2023). Definition 3.6 (p-precision Floating-Point Numbers (Merrill & Sabharwal, 2023)). The set of p-precision floating-point numbers Dp is defined as the collection of p-bit numbers, Dp = {0, 1}
p, including special values such as +inf, −inf, and nan. The set Dp can be naturally extended to vectors Dp
∗.

When p happens to be a finite number, we can also define the operations over p-precision float since the cardinality of the mappings between Dp vectors of m-dimension become at most finite (= 2pm·2 pm).

Definition 3.7 (Merrill & Sabharwal (2023)). A function f : Dp m → Dp nis a p-precision floating-point function if f can be computed by a p-space-bounded Turing machine.

The order and basic operations, including addition, subtraction, multiplication, and division, as well as operations involving special values (+inf, −inf, nan), follow the IEEE 754 standard (iee, 2019). The precision p can be defined as a function of the input sequence length n, determining the scale of precision as follows:
- Constant Precision: When p(n) is a constant function
(p(n) ∈ O(1)), the precision is fixed for any length of input.

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164
- Logarithmic Precision: When p(n) is a logarithmic function (p(n) ∈ O(log n)), the precision scales logarithmically with the input length.

In this work, our concern is constant precision. In the case of constant precision, p can be treated as a constant p ∈ N.

## 4. Transformer Decoder

This section introduces the mathematical and theoretic foundations of the Transformer Decoder Model, emphasizing its functional behavior (Def. 4.1), autoregressive capabilities (Def. 4.2), and alignment with formal languages (Def. 4.3).

## 4.1. Transformer Decoder

We focus on the decoder-based GPT (generative pretrained transformer) architecture (Radford et al., 2018). Unlike the original implementation, positional encoding (PE; details in § 4.1) is excluded in § 5 to facilitate theoretical analysis, while it is included in § 6 to reflect practical settings and evaluate its impact. In this work, all computations within the Transformer are conducted over the p-precision float numbers Dp (see § 3.2 and Merrill & Sabharwal (2023)). This constraint reflects a practical adaptation to real-world computational limits. Vocabulary space The vocabulary space of Transformers Σ∪V comprises the alphabet Σ and a set of special tokens V
(e.g., ⟨bos⟩, ⟨eos⟩, ⟨sep⟩). Special tokens are excluded from formal language, and there is no intersection to alphabet in this study. Basic string operations, such as concatenation, closure, and length, are defined over vocabulary spaces in the standard manner.

Transformer as a function Then we formalize the Transformer Decoder model as a function. Definition 4.1. A Transformer Decoder over p(n)-precision with parameters θ ∈ Params is a function:

$$\mathrm{TDec}_{p}(\cdot;\theta):(\Sigma\cup\mathbb{V})^{*}\to\Sigma\cup\mathbb{V}$$

where Σ ∪ V is the vocabulary space. Params represents the class of trainable parameters set, all components of the model. p(n) determines the internal precision depend on the input sequence length n.

Given an input sequence w1:n ∈ (Σ ∪ V)
∗, the Transformer Decoder outputs a single next token wn+1 =
TDecp(n)(w1:n; θ) ∈ Σ∪V, conditioned on the prefix w1:n and a set of parameter θ. Based on the formal definition of TDec, the computational flow from input to output generally follows the GPT model (Radford et al., 2018; Brown et al., 2020).

Positional encoding Since encoder-only Transformer cannot recognize the position of character, they need additional positional information. We denote abusolute positional encoding (APE) like Vaswani et al. (2017)'s sinusoidal one in this work. On the other hand, there are relative ones like T5 relative PE (Raffel et al., 2020) or ALiBi (Press et al., 2022). Alternatively, Kazemnejad et al. (2024) showed No Positional Encoding (NoPE) has good ability to generalize. In this work, we employ only APE and NoPE.

## 4.2. Autoregressive Token Generation

The Transformer Decoder model generates sentences autoregressively, predicting each token based on the input sequence and previously generated tokens until output a kind of end-of-sentence tokens. This process is formalized as follows: Definition 4.2. The t-times autoregressive composition (generation) of the Transformer Decoder function TDecp(·; θ) is denoted as TDecp t(·; θ) : (Σ∪V)
∗ → Σ∪V
and is recursively defined as:

$$\operatorname{TDec}_{p}{}^{t}(\sigma;\theta)={\begin{cases}\operatorname{TDec}_{p}(\sigma\cdot\operatorname{TDec}_{p}{}^{t-1}(\sigma;\theta);\theta)\\ \operatorname{TDec}_{p}(\sigma;\theta)\\ \operatorname{TDec}_{p}(\sigma;\theta)\end{cases}}({\mathrm{if}}\;t>1)$$

where · denotes token concatenation over Σ ∪ V.

This definition highlights the iterative nature of autoregressive generation, Furthermore, by restricting the codomain to the last token, this formulation aligns with the objectives of this study, emphasizing the relationship between autoregressive behavior and formal language recognition. From now on, when the context is clear, we simply write TDec.

## 4.3. **The Language Recognized By Transformer Decoder**

We now define the language recognized by a p-precision Transformer Decoder with a certain parameter θ and t-times decode steps, based on the definition 4.1 and 4.2.

Definition 4.3. The language recognized by such a t-times autoregressive transformer decoder with a certain parameters θ over p-precision, TDecp(·; θ), L(TDecp(·; θ)) is defined as:

$$L(\text{TDec}_{p}^{\;\;t}(\cdot;\theta),F)\tag{1}$$ $$=\{w\in\Sigma^{*}\,|\,\exists r\leq t(|w|).\text{TDec}_{p}^{\;\;r}(w\cdot\langle\text{sep}\rangle;\theta)\in F\}$$

where F ⊆ V is the nonempty set of accept token. Typically, F may include tokens such as ⟨eos⟩ or other special markers representing accept tokens.

Definition 4.3 states that an input string w is accepted if the output sequence TDec(w · ⟨sep⟩) ∈ (Σ ∪ V)
∗contains at least one accept token from F, within t(|w|) times or less autoregression. It is important to note that the special token ⟨sep⟩ is explicitly appended to the input sequence to distinguish the decoding sequence. Additionally, the length of the output sequence increase by a time function t : N → N, which maps the input sequence w to a maximum allowable number of decoding steps. For example: If t(n) = n 2, polynomially many decoding steps are permitted. If t(n) = c, decoding is restricted to a constant steps, regardless of the input length. Example 4.4. *Let the time function be a constant function* t(n) = 4, and the set of accept tokens be F = {⟨eos⟩}. Given that the output sequences of TDec for the input sequences "aabb" and "aa*" are as follows:*

$$\begin{array}{r l}{\operatorname{TDec}(a a b b(\operatorname{sep}))=a b a\langle\operatorname{eos}\rangle\ldots}\\ {\operatorname{TDec}(a a\langle\operatorname{sep}\rangle)=a a a a a\ldots}\end{array}$$

In this case, the Transformer accepts only "*aabb*".

## 4.4. Confirmation Of Constraints

All other hyperparameters, such as the number of layers L ≥ 2, the model dimension d, and attention heads, are fixed as O(1), regardless of the input sequence length n. In summary, this study incorporates certain modifications:
- *Exclusion of positional encoding (NoPE; only for* § 5) - Two-layer Transformer Block, Single-head Attention without pre-norm configuration (§ 4.1)
- Causal masking for attention computation, and softmax function within the Attention mechanism (§ 4.1)
- Greedy Search decoding (Definition 4.3)
This formalization bridges the autoregressive generation mechanism with the theoretical analysis of language recognition. In subsequent sections, we explore the expressivity of Transformer Decoder models within this framework.

## 5. **Main Result 1: Finiteness Of Fixed-Precision** Transformer Without Pe

In this section, we present our first main result concerning the expressivity of Transformers under fixed-precision arithmetic and softmax-based attention mechanisms. This result establishes a direct correspondence between the class of languages recognized by Transformers and finite or cofinite languages (Theorem 5.2) under a natural assumption (Assumption 5.1). Infinity-Free Parameter Assumption We begin by introducing a natural assumption regarding the parameters of the attention layers in Transformers.

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 Assumption 5.1 (Infinity-Freeness). For each attention layer, the matrix product of query and key vectors is always greater than minus infinity (−inf ∈ Dp):

$$\forall y,y^{\prime}\in\mathbb{D}_{p}^{\;\;d}.\quad Q(y)K(y^{\prime})^{\mathsf{T}}\neq\pm\mathrm{inf}$$
T̸= ±inf (2)
where d ∈ N is the model dimension, and *Q, K* : Dp d →
Dp dare the query and key affine transformations.

This assumption depends only on the parameters of the query and key affine transformations. It generally holds for most trained Transformer models. Theorem 5.2 (Finiteness and Co-finiteness of Languages Recognized by Transformer Decoder). Assume that Assumption 5.1 holds. Under this assumption, the languages recognized by any Transformer decoders is exactly finite or co-finite languages. Specifically, the following two statements hold:
1. **(upper bound)** For any p ∈ N, t(n) ∈ Ω(1), θ ∈
Params, F ⊆ V, there exists a finite or co-finite language Lf *such that* L(TDec, F) = Lf .

2. **(lower bound)** *For any finite or co-finite language* L
′ f
, there exist parameters p
′ ∈ N, t′(n) ∈ Ω(1), θ′ ∈
Params, F′ ⊆ V *such that* L
′ f 
= L(TDec, F′).

Theorem 5.2 represents a key result of this study. It states that, under the infinity-freeness (Assumption 5.1) and with fixed precision p, the class of languages recognized by Transformer decoders aligns exactly with the class of finite and co-finite languages, regardless of the specific parameters, the number of decoding steps, or the set of accept states. Or vice versa, that means when the input length exceeds a certain number, transformer model cannot distinguish the inputs. The two claims of Theorem 5.2 are proved in § 5.1 and § 5.2, respectively.

5.1. Proof of the Upper Bound under Assumption 5.1 Lemma 5.3. Suppose Assumption 5.1 holds. Then there exists an integer L ∈ N with the following property:
For any two inputs *w, w*′ ∈ Σ
∗ *with* |w|, |w
′| ≥ L, the Transformer decoder TDecp tw · ⟨sep⟩; θ*produces the* same *output tokens as it does for* TDecp tw
′· ⟨sep⟩; θ, provided w and w
′*share the same final character.*
Proof. Let us denote the final token of the input as v ∈ Σ ∪ {⟨sep⟩}. By Assumption 5.1, we know that for any vectors y, y′ ∈ (Dp)
d, the dot-product Q(y) K(y
′)
⊤ ̸= −inf.

In particular, we can choose constants α, β in (Dp)
d(related to the embedding of v) such that the repeated sum of exp(Q(α) K(β)
⊤) over enough positions saturates the p-precision range to +inf. Hence we define L to be the minimum length at which this "+inf sum" occurs in the causal masking scenario. Let w be any string with |w| ≥ L. When the decoder at time-step r attends over all previously seen tokens (⟨sep⟩ appended at the end), the *softmax denominator* in Attn(qv, Kw, Vw) accumulates

$$\sum_{j=1}^{|w|+1}\exp\bigl(q_{v}\,K_{j}^{\top}\bigr)$$

and by the definition of L, this sum diverges to +inf in p-precision. Consequently, the fraction expqv K⊤
|w|
/ (+inf) is effectively 0 in pprecision,making the final token's contribution vanish. Repeating this for each layer (and for each of the t(|w|) auto-regressive decoding steps) shows that any distinct differences in w vs. w
′(*provided* their last character is the same) are overshadowed as |w*| → ∞*.

Thus if *w, w*′ ∈ Σ
∗ both have length ≥ L and share the same last symbol v, the decoder outputs TDect(w · ⟨sep⟩; θ) and TDect(w
′·⟨sep⟩; θ) coincide. In other words, once the input length is beyond L, the model cannot further distinguish among long strings ending in the same symbol. Why Lemma 5.3 **implies finite/co-finite recognition.** By Lemma 5.3, all strings of length ≥ L that share a final character are mapped to the *same* sequence of output tokens under the t(|w|)-step decoding. Hence, if for some long string w the decoder *accepts* (i.e. produces a token in F ⊆
V), then *all sufficiently long strings* with the same last letter are also accepted. Thus we obtain either:
- A *co-finite* pattern: the model rejects only finitely many strings (those of length < L, plus possibly a few last-letter classes among the long strings), so L(TDectp
(·; θ), F) is co-finite.

- A *finite* pattern: the model accepts only finitely many cases (if it rejects all length ≥ L strings except perhaps a handful).

In both cases, the recognized language is either finite or co-finite.

Remark on ⟨sep⟩. Including a special terminal token ⟨sep⟩ in the input helps ensure that the "last symbol" alignment is explicit. Without it, one might rely on actual last letters in Σ, and the argument becomes a suffix-based distinction rather than a crisp boundary. Our Definition 4.3 ensures that w⟨sep⟩ standardizes the final token (or the last letter in w if no ⟨sep⟩ is appended),leading to a simpler classification at large lengths.

220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 5.2. Proof of the Lower Bound under Assumption 5.1 In this subsection, we show that a Transformer decoder can recognize any finite or co-finite language Lf ⊆ Σ
∗in constant (one or two) decoding steps. Formally, we will construct a p-precision Transformer decoder that outputs a special "accept" token (e.g. ⟨eos⟩) if and only if the input string belongs to Lf .

Lemma 5.4. Let L*f in* ⊆ Σ
∗ be any finite language. Then there exist: a precision parameter p ∈ N*, a parameter set* θ ∈ Params, and a set of accept state tokens F ⊆ V*, such* that the decoder recognizes Lf in *in exactly* one decoding step. That is, L(TDec, F) = L*f in*.

Proof. We design a two-layer Transformer decoder that first (i) accumulates sufficient information (e.g. a partial sum or isomorphic encoding of the entire input string), and then (ii) employs a feed-forward network (FFN) to map that information to a binary output: namely "w ∈ L" or
"w /∈ L". When w ∈ L*f in*, the decoder emits a special token (⟨eos⟩) on the single decoding step; otherwise it does not. Embedding layer. Suppose the input tokens are w1*, . . . , w*n ∈ Σ. Let p be large enough to accommodate all numerics (we will specify p in a moment). For each token wi, define its embedding vector as

$$\mathbf{x}_{i}:=[\,0,\mathrm{emb}(w_{i})\,]\in\mathbb{D}_{p}{}^{d}$$  where $\mathrm{emb}(w_{i})\in\mathbb{D}_{p}{}^{d-1}$.  
$$({\mathfrak{I}})$$

The extra leading coordinate (0) will be used to store positions or partial sums in subsequent layers. First attention layer. We apply a *uniform attention* to gather position-related or partial-sum information. For instance, let the query, key, and value transformations be:

$$\begin{array}{r l}{Q(\mathbf{x})=\mathbf{1},}&{{}K(\mathbf{x})=\mathbf{1},}\\ {V(\mathbf{x})=[\,1,0,\ldots,0\,]\in\mathbb{D}_{p}^{\;\;d}}\end{array}$$

for all input vectors x. Then, under causal masking (each xi only attends to x1:i), the attention output for xiis:

$$\operatorname{Attn}\!\left(Q(\mathbf{x}_{i}),K(\mathbf{x}_{1:i}),V(\mathbf{x}_{1:i})\right)={\Big[}{\frac{1}{i}},\,0,\ldots,0{\Big]}.$$
i. (5)
Thus, after adding the residual connection, the layer output becomes:

$\mathbf{a}_{i}^{1}=\left[\frac{1}{i},\,0,\,.\,.\,.\,,0\right]+\mathbf{x}_{i}=\left[\frac{1}{i},\,\mathrm{emb}(w_{i})\right]$.  
First feed-forward network. We design an FFN so that

$$\mathbf{z}_{i}^{1}=\mathrm{FFN}\!\left(\mathbf{a}_{i}^{1}\right)={\left[\begin{array}{l l}{0,}&{{\frac{\mathrm{emb}(w_{i})}{i}}}\end{array}\right]}$$
i(7)
$$(T)$$

This step ensures each position's embedding is scaled by 1/i and placed in the tail part of the vector.

Second attention layer. We next use the n-th token xn (or similarly the "final step") to attend over all z 11*, . . . ,* z 1n. Let Q(x) = 1, K(x) = 1, V (x) = x. Hence,

$$\begin{array}{r}{\mathbf{a}_{n}^{2}=\operatorname{Attn}\!\left(Q(\mathbf{z}_{n}^{1}),K(\mathbf{z}_{1:n}^{1}),V(\mathbf{z}_{1:n}^{1})\right)}\\ {={\frac{1}{n}}{\bigg[}0,\sum_{k=1}^{n}{\frac{\operatorname{emb}(w_{k})}{k}}{\bigg]}}\end{array}$$
$$(8)$$
i(8)
Since we choose p large enough, 1n
̸=p 0 in the p-precision sense.

Remark. The partial sum Pn k=1 emb(wk)
kcan be seen as carrying isomorphic information about (w1*, . . . , w*n), assuming a suitable injection or universal approximation property (we treat details abstractly here). Second feed-forward network. Finally, we use a universalapproximation argument: there is an MLP or FFN that can decode a 2n ∼ w1:n and output 1 iff w ∈ L*f in*:

$$\text{FFN}^{2}\left(\mathbf{a}_{n}^{2}\right)=\begin{cases}1&\text{(if$w\in L_{fin}$),}\\ 0&\text{(otherwise).}\end{cases}\tag{9}$$

We then interpret output "1" as a special accept token (e.g. ⟨eos⟩) in the output layer. Hence, the entire decoder recognizes exactly L*f in*

$$\left({\mathcal{A}}\right)$$

Extension to co-finite languages. For a co-finite language L*cof in*, we simply invert the behavior: almost all strings map to "1" (accept), while the finite exceptional set Σ
∗ \ L*cof in* maps to "0." A parallel argument with slight modifications (where the second FFN outputs 1 for nearly all inputs, except a finite listed set of strings) completes the proof.

$$({\boldsymbol{5}})$$

Conclusion. By combining these constructions, we see that any finite or co-finite language Lf ⊆ Σ
∗can indeed be recognized by a p-precision, two-layer Transformer decoder in one or two decoding steps. Thus, for such Lf , we have Lf = L(TDec) for some parameter choice and constant decode budget.

$$(6)$$

In summary, Assumption 5.1 plus the no-positionalencoding policy forces the Transformer decoder to unify all sufficiently long strings with identical trailing tokens. Hence the language recognized cannot exceed finite or co-finite sets, completing the proof of the upper bound.

6. Main Result 2: The language recognized by fixed-precision decoder 6.1. Lower Bound for asm. 5.1 **and APE** We now prove that a Transformer decoder with Assumption 5.1 *and some APE* can recognize any *cyclic language*.

Theorem 6.1. For any m-cyclic language Lc, there exist some Transformer with asm. 5.1 *and Abusolute Positional* Encoding, TDec′*such that* Lc = L(TDec′, Fc) for some set of special tokens Fc ⊆ V
Proof Sketch. **APE to distinguish positions mod** m For given m-cyclic language, prepare suitable APE such that have periodicity (e.g., sinusoidal embeddings (Vaswani et al., 2017; Chiang et al., 2023)), and the Transformer can effectively identify each position's residue class modulo m Hence, for index i and j*, if* i ≡ j (mod m), their positional encodings can be made same so that the network recognizes the same residue class. Attention mechanism In other words, for the head corresponding to residue r, only tokens xi with i ≡ r (mod m)
receive a high attention score. Under the "inf-free" condition, no key–query product becomes −inf, so we can rely on softmax-based attention to highlight precisely those tokens that belong to the right residue class.

FFN to implement the m**-cyclic condition** Since an mcyclic language determines acceptance based on how symbols appear in these residue classes, the final feed-forward network can be crafted to check the patterns aggregated from each class. Concretely, if Lc says "Positions ≡ r (mod m)
must contain letter a" or "must exclude letter b," then after the multi-head attention, the hidden representation has sufficient information to confirm or deny these constraints. Putting it all together, the Transformer obtains positionresidue awareness from APE, employs attention to gather all tokens of each residue class, and checks with a FFN whether the cyclic criteria are satisfied. Thus, for any mcyclic language Lc, we construct a suitable Transformer decoder (satisfying inf-free and using APE) so that Lc is recognized exactly by that model, completing the proof.

## 6.2. Lower Bound For Nope: Letter-Set Languages

We now prove that a Transformer decoder without any positional encoding (NoPE) and Assumption 5.1 can recognize any *letter-set language*. Formally, a *letter-set language* LS ⊆ Σ
∗is one where acceptance only depends on which letters (symbols) appear in w (not on their order or count). Theorem 6.2. For any letter-set language LS, there exist some Transformer decoders with No Positional Encoding and without Assumption *5.1,* TDec′′ *such that* Ls = L(TDec′′, Fs) *for some set of special tokens* Fs ⊆ V
330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 Proof Sketch. A letter-set language is determined solely by the set of unique letters in the input string. In a Transformer decoder without positional encoding, identical letters are mapped to identical embedding vectors, irrespective of their positions in the input sequence. Consequently, the model cannot distinguish whether a appears as the first or fifth letter, but it can identify whether a is present in the input at all. By processing embeddings, the model can determine the existence of each letter without tracking its count or position. Using attention and feed-forward layers, the model can consolidate these embeddings to produce a "presence flag" for each letter. The flag is set based on whether the embedding is zero or non-zero. Thus, we employ the non-finite floating-point value inf in the denominator during the transition computation to make the flags zero aligning with the discussion in Lamma 5.3. This is why the Assumption 5.1 is removed. For example, if a appears anywhere in the sequence, a specific hidden vector state can be activated to indicate its presence.

Since letter-set languages are defined by finite logical combinations of conditions on letter presence, the final feedforward and output layers can evaluate these conditions. For example, the model can output an accept token if the presence flags match the required subset S, or reject otherwise. This process effectively ignores order and frequency, focusing solely on whether each required letter is present at least once. The lack of positional encoding aligns naturally with the requirements of letter-set languages. A NoPE Transformer focuses on whether a given letter appears, without being influenced by order or frequency. Even if a letter a appears multiple times, the model only needs a single bit of information ("a exists") to make its decision. By aggregating these presence flags, the Transformer can determine whether the input satisfies the rules of the letter-set language. Thus, a NoPE Transformer can recognize any letter-set language, using its ability to abstract away positional information and focus on the presence of letters.

## 7. Discussions 7.1. What Is The Key Module In Transformers?

Although numerous studies have advanced our understanding of Transformers, a fundamental question remains:
"Which architectural component primarily contributes to their expressivity?" Despite extensive research on elements like attention mechanisms, layer normalization, and embedding schemes, there is no universal consensus on what exactly determines a language model's ability to capture complex linguistic phenomena.

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 Bhattamishra et al. (2020b) focused on the Turingcompleteness and the necessity of various architectural components and highlighted the crucial role of residual connections in maintaining expressivity. They also demonstrated that Transformers without explicit positional encoding but with positional masking remain Turing-complete. Similarly, Chiang et al. (2023) highlighted the importance of numerical precision (fixed vs arbitrary) and showed that the expressivity of such an encoder Transformer can be tightly upper-bounded by the language class FOC[+; MOD], a firstorder logic with counting quantifiers, addition, and modular arithmetic. A crucial difference emerges when comparing their results to ours: they included *positional encoding* (specifically a sinusoidal scheme), which allowed the model to handle periodic information effectively. In this study, We adopted a constant precision scheme similar to Chiang et al. (2023). Moreover, we introduced a reasonable practical Assumption 5.1. Building on these settings,we identified a Transformer setup capable of recognizing the minimal language, namely finite or co-finite languages, without any positional encodings (§ 5). This setup closely resembles real-world Transformers, leading us to hypothesize that practical Transformers may inherently be restricted to recognizing finite languages, functioning as highly efficient lookup tables. While prior studies (Bhattamishra et al., 2020b; Kazemnejad et al., 2024)) demonstrated the practical effectiveness of NoPE, our theoretical analysis suggests that NoPE has inherent limitations in enhancing expressivity. Next, we examined how adding absolute positional encoding (APE) and removing the assumption affected the tendency to restrict recognition to finite languages (§ 6). However, even with these additions, expressivity increased only slightly, as fixed-precision still constrains expressivity to near-finiteness. Our findings show that restricting precision from logarithmic (Merrill & Sabharwal, 2024b) to constant results in a significant loss of expressiveness. Furthermore, this loss increases as the number of decoding iterations grows, noting that expressivity reaches P when t ∈ O(n c).

## 7.2. Languge Modeling

Throughout this work, we frame the Transformer as a language recognizer, addressing the membership problem in a more formal sense rather than as a *language generator*. In practice, particularly in language modeling, a decoderbased Transformer typically produces tokens probabilistically, generating text rather than deciding membership in a formal language. In fact, research on the expressivity of language modeling exists (Svete & Cotterell, 2024; Nowak et al., 2024). While our "recognizer" viewpoint diverges somewhat from typical usage, bridging these two outlooks more rigorously remains a key objective for future research.

## 7.3. Potential Extensions

We acknowledge that our current setup is simplified, focusing on a limited subset of Transformer components: attention masking, the absence of layer normalization, and no extensive multi-head or multi-layer structure. In realworld architectures, additional architectural features could significantly impact expressivity. Furthermore, we have identified gaps (Table 1). A natural extension involves clarifying how these additional mechanisms, such as relative positional encoding or the softmax-tohardmax transition, might shift the upper and lower bounds on expressivity. We believe our fundamental approach can be adapted to investigate such enhancements, while leaving precise formalization and empirical validation for future work.

## 8. Conclusion

In this work, we examined the expressivity of fixedprecision Transformers to investigate their practical implications. To achieve this, we introduced three constraints: fixed-precision floating-point arithmetic, a reasonable assumption 5.1 regarding query-key parameters, and the presence or absence of positional encoding. In § 5, we demonstrated that Transformers operating under the constraints (Fixed-precision + Assumption 5.1 + NoPE) can recognize only finite or co-finite languages. In § 6, we further proved the role of Assumption 5.1 and also positional encoding, as relaxing either of these constraints slightly enhances expressivity. These findings suggest that these constraints impose fundamental limitations on Transformer expressivity. Future research could extend this analysis to language modeling or investigate how alternative modules and hardmax replacements influence expressivity.

## References

Ieee standard for floating-point arithmetic. IEEE Std 7542019 (Revision of IEEE 754-2008), pp. 1–84, 2019. doi: 10.1109/IEEESTD.2019.8766229.

Ackerman, J. and Cybenko, G. A survey of neural networks and formal languages, 2020. URL https://arxiv. org/abs/2006.01338.

Bhattamishra, S., Ahuja, K., and Goyal, N. On the Ability and Limitations of Transformers to Recognize Formal Languages. In Webber, B., Cohn, T., He, Y., and Liu, Y.

(eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 7096–7116, Online, November 2020a. Association for Computational Linguistics. doi: 10.18653/v1/2020.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 emnlp-main.576. URL https://aclanthology. org/2020.emnlp-main.576/.

Bhattamishra, S., Patel, A., and Goyal, N. On the computational power of transformers and its implications in sequence modeling. In Fernandez, R. and ´ Linzen, T. (eds.), Proceedings of the 24th Conference on Computational Natural Language Learning, pp. 455– 475, Online, November 2020b. Association for Computational Linguistics. doi: 10.18653/v1/2020.conll-1. 37. URL https://aclanthology.org/2020. conll-1.37/.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS '20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

Chiang, D. and Cholak, P. Overcoming a theoretical limitation of self-attention. In Muresan, S., Nakov, P., and Villavicencio, A. (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 7654–7664, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.527. URL https: //aclanthology.org/2022.acl-long.527/.

Chiang, D., Cholak, P., and Pillay, A. Tighter bounds on the expressivity of transformer encoders. In Proceedings of the 40th International Conference on Machine Learning, ICML'23. JMLR.org, 2023.

Chiang, D., Rawski, J., Strobl, L., and Yang, A.

Esslli 2024, 2024. https://sleynas.com/ esslli-2024-summer-school-course (202501 viewed).

Cybenko, G. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals, and Systems (MCSS), 2(4):303–314, December 1989. ISSN 0932-4194. doi: 10.1007/BF02551274. URL http: //dx.doi.org/10.1007/BF02551274.

Feng, G., Zhang, B., Gu, Y., Ye, H., He, D., and Wang, L.

Towards revealing the mystery behind chain of thought: a theoretical perspective. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS '23, Red Hook, NY, USA, 2023. Curran Associates Inc.

Hahn, M. Theoretical limitations of self-attention in neural sequence models. Transactions of the Association for Computational Linguistics, 8:156–171, 01 2020. ISSN 2307-387X. doi: 10.1162/tacl a 00306. URL https: //doi.org/10.1162/tacl_a_00306.

Hao, Y., Angluin, D., and Frank, R. Formal language recognition by hard attention transformers: Perspectives from circuit complexity. Transactions of the Association for Computational Linguistics, 10:800–810, 07 2022. ISSN 2307-387X. doi: 10.1162/tacl a 00490. URL https://doi.org/10.1162/tacl_a_00490.

Hornik, K., Stinchcombe, M., and White, H. Multilayer feedforward networks are universal approximators. Neural Networks, 2(5):359–366, 1989. ISSN 0893-6080. doi: https://doi.org/10.1016/0893-6080(89)90020-8. URL https://www.sciencedirect.com/ science/article/pii/0893608089900208.

Kajitsuka, T. and Sato, I. Are transformers with one layer self-attention using low-rank weight matrices universal approximators?, 2024. URL https://arxiv.org/
abs/2307.14023.

Kazemnejad, A., Padhi, I., Ramamurthy, K. N., Das, P.,
and Reddy, S. The impact of positional encoding on length generalization in transformers. In *Proceedings of* the 37th International Conference on Neural Information Processing Systems, NIPS '23, Red Hook, NY, USA, 2024. Curran Associates Inc.

Merrill, W. Formal language theory meets modern nlp, 2021.

URL https://arxiv.org/abs/2102.10094.

Merrill, W. Formal languages and the nlp black box.

In Developments in Language Theory: 27th International Conference, DLT 2023, Umea, Sweden, June ˚
12–16, 2023, Proceedings, pp. 1–8, Berlin, Heidelberg, 2023. Springer-Verlag. ISBN 978-3-031-33263-0. doi:
10.1007/978-3-031-33264-7 1. URL https://doi. org/10.1007/978-3-031-33264-7_1.

Merrill, W. and Sabharwal, A. The parallelism tradeoff:
Limitations of log-precision transformers. Transactions of the Association for Computational Linguistics, 11:531–
545, 2023. doi: 10.1162/tacl a 00562. URL https: //aclanthology.org/2023.tacl-1.31/.

Merrill, W. and Sabharwal, A. A logic for expressing logprecision transformers. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS '23, Red Hook, NY, USA, 2024a. Curran Associates Inc.

Merrill, W. and Sabharwal, A. The expressive power of transformers with chain of thought, 2024b. URL https:
//arxiv.org/abs/2310.07923.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Merrill, W., Sabharwal, A., and Smith, N. A. Saturated transformers are constant-depth threshold circuits. Transactions of the Association for Computational Linguistics, 10:843–856, 08 2022. ISSN 2307-387X. doi: 10.1162/tacl a 00493. URL https://doi.org/10. 1162/tacl_a_00493.

Nowak, F., Svete, A., Butoi, A., and Cotterell, R. On the representational capacity of neural language models with chain-of-thought reasoning. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), *Proceedings of the 62nd* Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12510–12548, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.

676. URL https://aclanthology.org/2024. acl-long.676/.

Park, S., Yun, C., Lee, J., and Shin, J. Minimum width for universal approximation, 2020. URL https:// arxiv.org/abs/2006.08859.

Press, O., Smith, N., and Lewis, M. Train short, test long:
Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations, 2022. URL https://openreview.net/
forum?id=R8sQPpGCv0.

Perez, J., Barcel ´ o, P., and Marinkovic, J. Attention is turing- ´
complete. *Journal of Machine Learning Research*, 22(75): 1–35, 2021. URL http://jmlr.org/papers/ v22/20-302.html.

Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. Improving language understanding by generative pretraining. 2018.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning* Research, 21(140):1–67, 2020. URL http://jmlr. org/papers/v21/20-074.html.

Strobl, L., Merrill, W., Weiss, G., Chiang, D., and Angluin, D. What formal languages can transformers express? a survey. Transactions of the Association for Computational Linguistics, 12:543–561, 2024. doi: 10. 1162/tacl a 00663. URL https://aclanthology. org/2024.tacl-1.30/.

Svete, A. and Cotterell, R. Transformers can represent n-gram language models. In Duh, K., Gomez, H., and Bethard, S. (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 6845–6881, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long. 381. URL https://aclanthology.org/2024. naacl-long.381/.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.,
2017. URL https://proceedings.neurips.

cc/paper_files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.

pdf.

Wei, C., Chen, Y., and Ma, T. Statistically meaningful approximation: a case study on approximating turing machines with transformers. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS '22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

Yang, A. and Chiang, D. Counting like transformers: Compiling temporal counting logic into softmax transformers, 2024. URL https://arxiv.org/abs/2404. 04393.

Yang, A., Chiang, D., and Angluin, D. Masked hardattention transformers recognize exactly the star-free languages, 2024. URL https://arxiv.org/abs/ 2310.13897.

Yao, S., Peng, B., Papadimitriou, C., and Narasimhan, K.

Self-attention networks can process bounded hierarchical languages. In Zong, C., Xia, F., Li, W., and Navigli, R. (eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 3770–3785, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.292. URL https: //aclanthology.org/2021.acl-long.292/.

Yun, C., Bhojanapalli, S., Rawat, A. S., Reddi, S. J., and Kumar, S. Are transformers universal approximators of sequence-to-sequence functions?, 2020. URL https: //arxiv.org/abs/1912.10077.