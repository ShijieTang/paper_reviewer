# Roll The Dice & Look Before You Leap:

Vaishnavh Nagarajan * 1 **Chen Henry Wu** * 2 Charles Ding 2 **Aditi Raghunathan** 2

## Abstract

We design a suite of minimal algorithmic tasks that are a loose abstraction of *open-ended* realworld tasks. This allows us to cleanly and controllably quantify the creative limits of the presentday language model. Much like real-world tasks that require a creative, far-sighted leap of thought, our tasks require an implicit, open-ended stochastic planning step that either (a) discovers new connections in an abstract knowledge graph (like in wordplay, drawing analogies, or research) or (b) constructs new patterns (like in designing math problems or new proteins). In these tasks, we empirically and conceptually argue how nexttoken learning is myopic; multi-token approaches, namely teacherless training and diffusion models, comparatively excel in producing diverse and original output. Secondly, to elicit randomness without hurting coherence, we find that injecting noise at the input layer (dubbed seedconditioning) works surprisingly as well as (and in some conditions, better than) temperature sampling from the output layer. Thus, our work offers a principled, minimal test-bed for analyzing openended creative skills, and offers new arguments for going beyond next-token learning and temperature sampling. We make part of the code available under https://github.com/chenwu98/ algorithmic-creativity

## 1. Introduction

surprising and fresh connections never seen before. For instance, consider responding to highly under-specified prompts like "Generate a challenging high-school word problem involving the Pythagoras Theorem." or "Suggest some candidate therapeutic antibodies targeting the HER2 antigen." or
"Provide a vivid analogy to differentiate quantum and classical mechanics." Creativity in such tasks requires generating responses that are not just correct or *coherent*, but are also *diverse* across responses and are *original* compared to the training data. These currently-sidelined desiderata will rise to prominence as we explore LLMs for open-ended scientific discovery (Gruver et al., 2023; Romera-Paredes et al., 2024; Hayes et al.,
2025), for generating novel training data (Yu et al., 2024; Yang et al., 2024c; Wang et al., 2023), and as we scale up test-time compute approaches that benefit from diversity in exploration, such as best-of-N (Cobbe et al., 2021; Chow et al., 2024; Dang et al., 2025) and long chain-of-thought reasoning (OpenAI, 2024; DeepSeek-AI, 2025; Snell et al., 2024; Wu et al., 2024). Unlike simple open-ended tasks like generating names and basic sentences (Zhang et al., 2024b; Hopkins et al., 2023; Bigelow et al., 2024), many creative tasks (like designing a clever Olympiad problem) are said to involve a random flash of creative insight termed variously as a *leap of thought* (Wang et al., 2024a; Talmor et al., 2020; Zhong et al., 2024), a "eureka" moment (Bubeck et al., 2023), a mental leap (Holyoak & Thagard, 1995; Callaway, 2013; Hofstadter, 1995) or an incubation step (Varshney et al., 2019). The thesis of this paper is that learning to solve such creative leap-of-thought tasks (defined shortly) is misaligned with the current language modeling paradigm (a) in terms of next-token learning, and (b) in how randomness is elicited. We articulate these two concerns by designing a suite of algorithmic tasks inspired by such creative tasks. We then demonstrate how the creativity of language models suffers in these tasks, and how this can be alleviated (to an extent, within our tasks). Concretely, for the scope of this paper, a creative leap-ofthought task refers to tasks that involve a search-and-plan process; crucially, this process orchestrates multiple *random* 1 decisions *in advance* before generating the output. Typically, such a leap of thought is *highly implicit* in the text - to infer it, one has to deeply engage with the text and detect higher-order patterns. We can think of tasks like designing satisfying math problems, generating worthwhile research ideas, or drawing surprising analogies as examples of such tasks. Ideally, one would directly study these real-world tasks to quantify the limits of language models. Indeed, a flurry of recent works report that LLM-generated research ideas tend to be rephrased from existing ideas (Gupta & Pruthi, 2025; Beel et al., 2025) and that LLM outputs tend to be less creative than humans e.g., Chakrabarty et al. (2024); Lu et al. (2024b) (See §K). While assessing real-world tasks is a lofty goal, the evaluations are subjective (Wang et al., 2024b; Runco & Jaeger, 2012), and when the model has been exposed to all of the internet, originality is hard to ascertain. Thus, the conclusions will inevitably invite debate
(such as Si et al. (2024) vs. Gupta & Pruthi (2025) or Lu et al. (2024a) vs. Beel et al. (2025)).

In search of more definitive conclusions, we approach from a different angle: we study controllable tasks that are loose abstractions of real-world tasks and yet allow one to objectively quantify originality and diversity. This follows recent works that have studied diversity of models in path-finding (Khona et al., 2024) and challenging context-free grammars (CFGs) (Allen-Zhu & Li, 2023b). We broadly term these as open-ended algorithmic tasks. Our aim then is design minimal instances of these tasks, akin to basic arithmetic for reasoning, so as to tease apart the bare minimum computational skills required for creativity. Then, we can examine how the current modeling paradigm may be limited even with this basic skills, and what alternative paradigms may be desired. As our first main contribution, we draw inspiration from cognitive science literature (Boden, 2003) (see also Franceschelli & Musolesi (2023)) to design algorithmic tasks isolating two distinct types of creative leaps of thought.

The first class of tasks involves *combinational creativity*:
drawing novel connections in knowledge, like in research, wordplay or drawing analogies (see Fig 1 for task description). The second class of tasks involves exploratory creativity: constructing fresh patterns subject to certain rules, like in designing problems and suspense (see Fig 2). In these tasks, we can precisely evaluate models for the fraction of generations that are *coherent, unique and original* (not present in training set). We term this metric "algorithmic creativity" to denote that it solely evaluates the computational aspects of creativity. Within this framework, we articulate two creative limits of the current language modeling paradigm. First, we empirically find that next-token learning achieves lower algo-

X
b Generate: "g, f, Y" 
(in-weights graph)
s.t.

c d f e g i h Y
Generate: "a, b, c" b a c s.t. d e f
(in-weights graph)
Figure 1. **Minimal tasks inspired by combinational creativity:** Skills like research, humor and analogies often require identifying novel multi-hop connections from *known* pair-wise relationships in a knowledge graph. For instance, creating the wordplay "What kind of shoes do spies wear? Sneakers." requires searching over a semantic graph, and carefully planning a pair of words (shoes, spies) that lead to a mutual neighbor
(sneakers). Inspired by this, we define tasks where a symbolic graph is stored in the model weights; the model is exposed to example node sequences that form a specific multi-hop structure (like a sibling or a triangle) during training. The model must infer this structure from training; during inference, the model must implicitly recall-search-and-plan to generate novel and diverse node sequences obeying the same structure in the in-weights graph. Pictured are two example tasks with a symbolic graph each, and a corresponding example sequence obeying a sibling (g, f, Y) or a triangle structure (a, b, c). More details in §2.3 and Fig 9.

s.t. 

e b d c a Generate:
Generate:
s.t.

a b c d e
"a→b, c→d, d→e, b→c, e→a"
"c→a, b→d, d→c, e→b"
rithmic creativity compared to some multi-token approach, namely, either teacherless training (Bachmann & Nagarajan, 2024; Monea et al., 2023; Tschannen et al., 2023) or diffusion (Hoogeboom et al., 2021; Austin et al., 2021; Lou et al., 2023) (see Fig 3 and Fig 4). Our argument is that in all our tasks, inferring the latent leap of thought requires observing global higher-order patterns rather than local next-token patterns in the sequence.

Next, we turn to how we elicit randomness from a Transformer. While the de facto approach is to elicit randomness

0.04 Training Objective:
Standard
(Next-Token)
Teacherless
(Multi-Token)
0.04 0.02 0.00 0.02 0.04 Line Construction 0.00 0.25 0.50 0.75 Creativity Triangle Discovery 0.0 0.5 1.0 Circle Construction 0.0 0.2 0.4 Sibling Discovery 0.0 0.2 0.4 Line Construction 0.0 0.1 0.2 0.3 Memorization Triangle Discovery 0.00 0.25 0.50 0.75 Sibling Discovery 0.00 0.25 0.50 0.75 Circle Construction 0.00 0.25 0.50 0.75
at the output - temperature sampling - we contrast this with injecting randomness into the model. Concretely, we study *seed-conditioning*, where we train and test with a random prefix string per example. Surprisingly, we find that seed-conditioning induces non-trivial algorithmic creativity even with deterministic decoding; it is in fact, comparable to temperature sampling (and in some cases, better). We intuit that maximizing diversity at the output-level results in cognitive overload: it requires simultaneously processing many leaps of thoughts to compute a marginalized token distribution. Input-level noise injection could sidestep this by articulating one leap of thought per seed. We also view this exploration as a controlled study solidifying prior indications that prompt variations induce diversity (Li et al., 2023; Lau et al., 2024; Naik et al., 2024; Li et al., 2022). Overall, we hope to advance the field in two directions. First, we provide a new angle to advocate for multi-token approaches, orthogonal to the "path-star" example in Bachmann & Nagarajan (2024) (or B&N'24 in short). Whereas, the path-star example portrays a gap in correctness of reasoning, ours shows a gap in diversity of open-ended thinking. We note though that B&N'24 is an impossibility result where next-token learning breaks down spectacularly (unless there is exponential data or compute), while ours is a data-inefficiency result (where next-token learning occurs but is mediocre). Next, the gap we show appears even in 2token-lookahead tasks as against the many-token-lookahead path-star task. Third, and most conceptually important is the fact that, while the path-star task is amenable to nexttoken prediction upon reversing the tokens, we identify tasks where no re-ordering is friendly towards next-token prediction - *the optimal thing to do is to globally learn* higher-order patterns implicit in the whole future sequence. This presents a challenge to recent proposals that aim to fix next-token prediction via permutations (Pannatier et al., 2024; Thankaraj et al., 2025) or partial lookaheads (Bavarian et al., 2022; Fried et al., 2022; Kitouni et al., 2024; Nolte et al., 2024). As a second direction of progress, we hope our work provides a foundation to think about open-ended tasks which are extremely hard to quantify in the wild. This may spur more algorithmic explorations on improving diversity (such as our approach of seed-conditioning) and on curbing verbatim memorization in language models.

## Our Contributions:

1. We create minimal, controlled and easy-to-quantify openended algorithmic tasks. These tasks isolate, and loosely capture two fundamental modes of creativity.

2. We find that multi-token prediction through one of teacherless training or diffusion, results in significantly increased algorithmic creativity in our tasks compared to next-token prediction.

3. Our argument provides new support for multi-token prediction, going beyond B&N'24. We show a gap in creativity in an open-ended task (rather than correctness in a deterministic one), in much simpler 2-token-lookahead tasks, and in tasks where no token permutation is friendly to next-token-learning.

4. We find that input-randomization via seed-conditioning achieves algorithmic creativity comparable with conventional output-randomization via temperature sampling.

## 2. Open-Ended Algorithmic Tasks & Two Types Of Creativity

We are interested in designing simple algorithmic tasks that are loosely inspired by endeavors such as generating scientific ideas, wordplay, narration, or problem-set design, where one needs to generate strings that are both "interesting" and never seen before. In all these tasks, before generating the output, one requires a (creative) leap of thought, a process that (a) is implicit i.e., is not spelled out in token space (or is even inherently hard to spell out), (b) involves discrete *random* choices (c) and together, those choices must be *coherent* in that they are carefully planned to satisfy various non-trivial, discrete constraints. These constraints fundamentally define the task and make it interesting e.g., a word problem should be solvable by arithmetic rules, or a pun must deliver a surprising punchline. The goal in such open-ended tasks is not just coherence though, but also diversity and novelty - generations must be as varied as possible and must not be regurgitated training data.

Before we design tasks that require the aforementioned leap of thought, we first clarify what tasks do not require it. Open-ended tasks that do not **require a leap of thought.** One simple open-ended task that may come to mind is generating uniformly-random known entities, like celebrity names (Zhang et al., 2024b). However, there is no opportunity to create a novel string here. A more interesting example may be generating grammatically coherent probabilistic CFG strings following a subject verb object format e.g., the cat chased a rat (Hopkins et al., 2023). While novel strings become possible now, no sophisticated leaps of thought are involved; each token can be generated on the fly, satisfying a local next-token constraint to be coherent. In light of this, we can rephrase our goal as designing open-ended, creative tasks where coherence requires satisfying more interesting, "global" constraints. To build this systematically, we draw inspiration from literature in cognitive science (Boden, 2003). Boden (2003) argues that fundamentally, there are three forms of creativity in that order: *combinational, exploratory* and *transformative*. We elaborate on the first two (the last, we do not look at).

## 2.1. The Fundamental Types Of Creativity (Boden, **2003)**

Combinational creativity. Consider rudimentary wordplay of the form "What musical genre do balloons enjoy? Pop music." or "What kind of shoes do spies wear? Sneakers." There is a global structure here: two unrelated entities (genre & balloons) are related eventually through a punchline (pop); the punchline itself is a mutual neighbor on a semantic graph. More broadly, Boden (2003) argues that many tasks, like the above, involve
"making unfamiliar combinations of familiar ideas" or the "unexpected juxtaposition of [known] ideas". Other tasks include drawing analogies, or finding connections between disparate ideas in science. All these tasks involve a leap of thought that in effect searches and plans over a space of known facts and combines them. Exploratory creativity. Consider the act of developing a mystery or designing logical puzzles. These endeavors are not as knowledge-heavy. What they crucially require is constructing fresh patterns that satisfy some highly nontrivial global constraint e.g., being resolvable as per some rules (e.g., logic). Such endeavors fall into a second class of *exploratory* creativity in Boden (2003). This includes much grander forms of exploration e.g., exploring various forms of outputs within a stylistic constraint, or exploring various corollaries within a theoretical paradigm in physics or chemistry. The leap of thought here requires searching over all possible sequences, constrained by a set of rules. In the upcoming sections, we will capture some core computations of rudimentary instances within the two creative skills above. By no means does this minimal algorithmic setup intend to capture the human values or the subjective aspects of these endeavors (Schmidhuber, 2009); nor do they capture the rich array of creative acts that Boden (2003) discusses within these categories. (See limitations in §A).

## 2.2. The Basic Setting And Notations

In all our tasks, we assume the standard generative model setting: the model must learn an underlying distribution D
through a training set S of m independent samples si ∼
D. The distribution is over a space V
L of L-length strings composed of tokens from the vocabulary V. The tasks are open-ended in that there is no one correct answer at test-time. The goal is to produce a random string from D, much like responding to the query Design a highschool word problem. We hope that this setup loosely resembles pretraining for language or in applications like drug discovery or protein (Meier et al., 2021; Madani et al.,
2023; Watson et al., 2023) and genome modeling (Rives et al., 2021; Nguyen et al., 2024), where the model is trained on a set of varied example sequences, and is later used to produce novel outputs during inference. Coherence: Each task is defined by a boolean coherence function coh : V
L *7→ {*true, false} which is true only on the support i.e., supp(D) = {s ∈ V
L| coh(s)}. The exact form of coh will be defined in each algorithmic task but broadly, we are interested in scenarios where determining coherence requires a global understanding of the whole string. This is inspired by the fact that a wordplay must have a preplanned punchline connecting what comes before, or a word problem must be solvable. We can think of D to be a simple uniform distribution over all coherent strings. Algorithmic creativity: Upon witnessing a finite set of examples, the model must learn to generate only strings that are (a) coherent, (b) original (not memorized) and (c) diverse (covers the whole support). An exact quantification of this is computationally expensive in our tasks. Instead, we approximate it by sampling a set T of many independent generations from the model and computing the fraction of T that is original, coherent and unique. Let the boolean memS(s) denote whether an example s is from the training set S and let the integer function uniq(X) denote the number of unique examples in a set X. (The exact definitions of these quantities vary by tasks, as we will see). Then, we define our (empirical) algorithmic creativity metric:

$$\hat{\bf cr}_{N}(T)=\frac{\bf uniq(\{s\in T|-mem_{S}(s)\wedge coh(s)\})}{|T|}.\tag{1}$$

Admittedly, we are looking at a simple form of novelty that is in-distribution. This notion, while simple, finds its relevance in applications like modeling proteins and genomes. We will also see that this notion is non-trivial enough to demarcate the limits of language models.

## 2.3. Tasks Inspired By Combinational Creativity

Combinational creativity requires a recall-and-search through entities from memory, constrained to relate with each other in an interesting way. We abstract this through tasks that discover structures from an *in-weights* graph i.e., a graph stored in the model weights, not reveal in context.

## 2.3.1. Sibling Discovery

This task involves an implicit, bipartite G with parent vertices V = {*A, B, C, . . .*} each neighboring a set of children nbr(A) = {a1, a2*, . . . ,* }, nbr(B) = {b1, b2*, . . .*} and so on. We define coh(s) to be true on sibling-parent triplets of the form s = (*γ, γ*0, Γ) such that *γ, γ*0 ∈ nbr(Γ). We then consider a uniform distribution D over all coherent strings (*γ, γ*0, Γ) for a fixed graph G. The model witnesses i.i.d samples from D. During test-time, the model must maximize algorithmic creativity (Eq 1) by generating novel parent-sibling triplets based on its in-weights knowledge of G. Note that the model is not provided the graph in-context as this would sidestep a core computational step in combinational creativity: recalling facts from a large memory (see §C.4). The hope is that the model infers and stores the pairwise adjacencies of G in its weights (given sufficient data). Full dataset description is in §D and Fig 9. We view this task as an abstraction of the wordplay example. One can think of the parent Γ as the "punchline" that delivers a connection between otherwise non-adjacent vertices, in the same way sneaker surprisingly connects the otherwise non-adjacent words, spies and shoes.

What is a leap of thought? The natural order of generation is to pick the parent vertex (i.e., punchline) first, and the siblings after (conditioned on the parent). Thus, if the task demanded producing (Γ*, γ, γ*0), the tokens we observe betray the way they are naturally generated. However, the wordplay example presents the punchline (the parent, Γ) last while implicitly it must be planned ahead. We term this implicit (random) planning step a leap of thought. Paralleling this, we define our sibling discovery task to generate the triplets as s = (*γ, γ*0, Γ), where the parent appears last. We hypothesize that this (sibling-first) construction is adversarial towards next-token learning, while a reversed (parent-first) dataset is friendlier towards next-token learning. (More on this in §2.6.)

## 2.3.2. Triangle Discovery

Next, we design a task that requires a more complex, higherorder planning: generating triangles from an appropriatelyconstructed knowledge graph G = (V, E) (see graph construction in §D). Thus, in this task coh((v1, v2, v3)) =
true iff all three edges between {v1, v2, v3} belong in G.

Furthermore, we define uniq(·) and mem(·) such that various permutations of the same triangle are counted as one (see details in §D, including the exact formatting of the string). Note that the leap of thought in this task is much harder to learn and execute as it requires co-ordinating three edges in parallel, from memory. This type of a higher-order planning task can be thought of an abstraction of more complex wordplay (like antanaclasis, where a word must repeat in two different senses in a sentence, while still being coherently related to the rest of the sentence) or creating word games (like crosswords) or discovering contradictions or feedback loops in a body of knowledge, an essential research skill - see §C.5.

## 2.4. Tasks Inspired By Exploratory Creativity

We are also interested in the less-knowledge-intensive creativity involved in tasks like designing word problems that demand novel solutions. We capture this skill minimally through tasks that construct new structures. No knowledge graph is involved in these tasks.

## 2.4.1. Circle Construction

In this task, the model must generate adjacency lists that can be rearranged to recover circle graphs of N vertices.

Let the generated list be s = (vi1, vi2),(vi3, vi4)*, . . .*. We define coh(s) = true iff there exists a *resolving* permutation π such that π(s) = (vj1
, vj2
),(vj2
, vj3
)*, . . .*(vjn
, vj1
)
for distinct j1, j2*, . . . j*n. i.e., each edge leads to the next, and eventually circles back to the first vertex. We define uniq and mem such that different examples with the same resolving π are counted as the same, even if they have differing vertices. As always, the learner is then exposed to a finite set of uniformly sampled coherent strings. Note that the latent leap of thought here requires constructing a novel permutation π before generating the sequence. Loosely, we can think of the resolving permutation π as how a conflict in a story or a word problem or a puzzle is solved; the vertices as characters or mathematical objects; the rules of rearranging an adjacency list as rules of logic, math or story-building. The creative goal in this task is to create novel dynamics in the conflict, or equivalently, novel dynamics in how the conflict is resolved. Thus if only the entities differ, but the plot dynamics remain unaltered, we count them as duplicates. See details in §D.

## 2.4.2. Line Construction

A simple variant of the above task is one where the edge list is of a line graph. The resolving permutation π satisfies π(s) = (vj1, vj2),(vj2, vj3). . . ,(vjn−1, vjn) for distinct j1 *. . . j*n. i.e., each edge leads to the next until a dead-end.

## 2.5. Leap Of Thought Is Obscured At The Token-Level

Rarely does human-written creative output come annotated with an articulation of the background creative process. Nor does each protein or molecule come with an enumeration of the laws that generated it - in fact, it is because those laws are not fully known do we delegate the task of inferring those to a machine. Likewise, in our setting, the training signal consists only of the observed tokens; the model must infer the underlying leap of thought. In fact, in our last three tasks, the model must infer a deeper form of implicit structure than what exists in many typical algorithmic tasks, like addition (Lee et al., 2024), or the pathstar task (B&N'24) or Sibling Discovery. Whereas in those tasks, the leap of thought is perceptible at the tokenlevel, modulo some re-ordering, our last three tasks have no such token ordering. These task are permutation-invariant
- no token is more privileged than the other, and all tokens must be "simultaneously learned" to infer the underlying process. We view this as an abstraction of tasks where the creative process is not immediate from the surface of the text. These tasks offer a test-bed even for non-next-token approaches that rely on re-permuting the tokens (Pannatier et al., 2024; Thankaraj et al., 2025) or predicting only a part of the future (Kitouni et al., 2024; Nolte et al., 2024; Bavarian et al., 2022; Fried et al., 2022).

## 2.6. How Next-Token Learning May Suffer In Our Tasks

Much like in sophisticated creative tasks, in our tasks, the most natural way to generate the string is by planning various random latent choices (say z) in advance and by producing a *plan-conditioned* distribution p(s|z) over coherent strings s. However, next-token prediction (NTP) - or nexttoken learning to be precise - we argue, is myopic and may struggle to learn such a latent plan. Our argument extends that of B&N'24 to our even simpler tasks. Consider the sibling task where we must generate siblingparent triplets (*γ, γ*0, Γ). The most natural generative rule is to plan the last token (the parent) first and decide the children last. Think of this as learning a latent plan z := Γ.

Then, learning the plan-conditioned generator p(*γ, γ*0, Γ|z)
factorizes to learning the distribution of children conditioned on a parent as p(γ|z := Γ) and p(γ 0|z := Γ) (due to conditional independence), and the trivial p(Γ|z := Γ). This requires only as many parent-sibling edges as there are in the graph, i.e., O(m · n) many points, if there are m parents, each with n children. This is optimal. Things should proceed differently with NTP. We argue that a NTP-learner would fail to learn the plan z := Γ. The key intuition is that an NTP-learner learns the parent Γ witnessing the siblings (*γ, γ*0) as input. This is trivial to fit: the parent is simply the mutual neighbor of the two siblings revealed in the prefix; B&N'24 term this a "Clever Hans cheat" since the model witnesses and exploits part of the ground-truth it must generate (the siblings). Such cheats, being simpler than even the true generative rule, are quickly picked up during learning due to the well-known simplicity bias of gradient-descent-trained neural networks (Shah et al., 2020). Subsequently, any gradient supervision from the parent Γ is lost (also known as gradient starvation (Pezeshki et al., 2021)), leaving no guidance to learn the latent plan, z := Γ. After the Clever Hans cheat is learned, we conjecture that the NTP-learner learns the second sibling not through the planconditioned distribution p(γ 0|z := Γ) but through the nexttoken-conditional, p(γ 0|γ). This is a complex distribution:
learning this may require witnessing every sibling-sibling pair totalling O(m · n 2) many training data - larger by a factor of n compared to the natural rule. Summary of argument. Abstractly, in a creative leap-ofthought task, it is most efficient to learn a well-planned random latent p(z) and a subsequent latent-conditioned distribution p(s|z). However, NTP factorizes this into pieces of the form p(si|s<i, z). Consequently, on the later tokens, the model may be lured by Clever Hans cheats and learn uninformative latents. Conversely, the model may learn the earlier through complex rules, bereft of a latent plan. While this may not lead to complete breakdown of learning as in B&N'24 in our tasks, we hypothesize it must lead to data-hungry learning.

## 3. Training And Inference

Transformers. For our next-token-trained (NTP) models, we use the standard teacher-forcing objective used in supervised finetuning. Given prompt p and ground truth sequence s, the model is trained to predict the i'th token si, given as input the prompt and all ground truth tokens up until that point, (p, s<i). We write the objective more explicitly in §B Eq 2. For the multi-token Transformer models, we use teacherless training (Monea et al., 2023; Bachmann & Nagarajan, 2024; Tschannen et al., 2023), where the model is trained to predict si simultaneously for all i, only given the prompt p (and some dummy tokens masking the s that was once given as input). Since the exact details of this is irrelevant to our discussion, we describe this in Eq 2. To train our models, we use a hybrid of this objective and the next-token objective. Diffusion models. We use the score entropy discrete diffusion model (SEDD (90M), Lou et al., 2023). Loosely, we can think of the objective here as a hybrid of the teacherless objective, where the input is fully masked or corrupted (and the model must unmask or uncorrupt all of it), and various easier objectives with partial maskings or corruptions of the input. During inference, the model starts with a fully masked or corrupted sequence of tokens, and iteratively corrects subsets of them. Inference. We extract *independent* samples from the model. For Transformers, during inference, we perform standard autoregression in both the next- and multi-token trained settings. We do this either with greedy decoding or with nucleus sampling (Finlayson et al., 2024).

## 3.1. Seed-Conditioning

While temperature sampling is the standard way to elicit diversity from a Transformer, we speculate that this can lead to cognitive overload: the model must process multiple thoughts to compute a diverse softmax distribution. Alternatively, we propose conditioning on a random seed as input to the model, hoping that a thought can be fleshed out with singular focus. Thus, during training, each point is associated with an arbitrary (meaningless) prefix (e.g., of random characters) called a seed (note that there is no prefix in our tasks besides the seed). During test-time, novel seeds are used to extract test data. We can also view seedconditioning more neutrally as a controlled way to simulate variations in prompt-wordings for a fixed task (e.g., design a word problem vs construct a problem). We elaborate on these intuitions in §C.1.

## 4. Experimental Results

Key details. Part of our experiments are performed for a Gemma v1 (2B) pre-trained model (Gemma Team et al., 2024), averaged over 4 runs. For diffusion, we use a 90M (non-embedding) parameters Score Entropy Discrete Diffusion model (SEDD (90M); Lou et al., 2023). For a fair comparison against NTP, we use a 86M (non-embedding) parameters GPT-2 (86M) model (Radford et al., 2019). In all our experiments, we finetune the models until it is clear that algorithmic creativity (Eq. 1) has saturated. All values are reported from this checkpoint. Finally, since our best Transformer results were under seed-conditioning (for both next- and mult-token training), our main results are reported under that training setting; we provide various ablations without that as well. Please see §E for more experimental details, and §D for precise dataset details (e.g., how the graph is constructed, how sequences are formatted etc.,).

## 4.1. Observations

Multi-token prediction improves algorithmic creativity significantly. In all our datasets, the algorithmic creativity of the Gemma v1 (2B) model increases significantly under multi-token prediction ( Fig 3), with nearly a 5x factor for the discovery datasets. Note that for this, we have selected the learning rate favorable towards next-token prediction; tuning for multi-token yields further gains (Fig 15). For 

Training Objective Standard (Next-Token) Teacherless (Multi-Token)
Diffusion-Absorb (Multi-Token) Diffusion-Uniform (Multi-Token)
Line Construction 0.0 0.5 1.0 Creativity Sibling Discovery 0.0 0.5 1.0 Triangle Discovery 0.00 0.05 0.10 Circle Construction 0.0 0.5 Memorization Circle Construction 0.0 0.1 0.2 Triangle Discovery 0.00 0.25 0.50 Sibling Discovery 0.00 0.05 Line Construction 0.00 0.02
the smaller models, in Fig 4, we find that the diffusion model improves over next-token training of similar-sized Transformers on all tasks, except on Sibling Discovery where it is mildly worse. With teacherless training, the gains are absent; however we find that the creativity of the top-K samples is generally improved (Fig 24), indicating a less peaky distribution. Regardless, the overall trend echoes prior findings that for smaller Transformers, teacherless training is hard to optimize and other multi-token objectives even hurt (Gloeckle et al., 2024) smaller models. Multi-token prediction reduces memorization for the larger model. Algorithmic creativity may suffer either due to either incoherence or mode collapse or memorization. For our larger model, it is the last reason that dominates: across the board (in Fig 3), next-token prediction is significantly prone to memorizing the data, while the multi-token method is highly resistant. As foreshadowed in §2.6, we hypothesize that this is because NTP memorizes the earlier training tokens without a global plan, having fit the later tokens via local coherence rules (because of Clever Hans cheats à la B&N'24). In contrast, the smaller models have much lower absolute memorization values; the multi-token objective preserves or mildly worsens this, in a way that does not always hurt creativity proportionally. here, we find that amongst the top-K samples, teacherless training has reduced memorization for some tasks (see Fig 25).

We point the reader to §C.6 for further empirical evidence supporting our argument about NTP from §2.6, including

0.04 0.02 0.00 0.02 0.04 0.04 0.02 Training Objective:Standard
(Next-Token)Teacherless
(Multi-Token)
Cr ea tiv ity SiblingDiscovery Cr ea tiv ity TriangleDiscovery seed10 greedy seed4 greedy seed2 temp2.0 null temp2.0 Inference Type 0.0 0.5 1.0 seed10 greedy seed4 greedy seed2 temp2.0 null temp2.0 Inference Type 0.0 0.2 0.4 Cre at iv ity CircleConstruction Creat ivi ty LineConstruction seed10 greedy seed4 greedy seed2 temp2.0 null temp2.0 Inference Type 0.0 0.2 0.4 seed10 greedy seed4 greedy seed2 temp2.0 null temp2.0 Inference Type 0.00 0.25 0.50 0.75

experiments on token-reordering and experiments ruling out other hypotheses. Seed-conditioning improves algorithmic creativity for Transformers. Orthogonal to the effect of multi-token vs. next-token objectives, we point out the effect of seedconditioning a Transformer. First, and most remarkably, with seed-conditioning, there is no need for temperature:
even deterministic greedy decoding generates diverse outputs and highly non-trivial creativity (Fig 5, Fig 6). This means that, even though the seeds we provide are arbitrarily paired with the training datapoint, the models have learned to transform the arbitrary noise into meaningful randomness. Next, increasing the seed lengths consistently boosts algorithmic creativity for both next-token and multi-token approaches (Fig 5, 23), presumably up to a limit. Thus, for Transformers, we propose viewing seed-conditioning as a knob for diversity distinct from temperature sampling. Next, seed-conditioning results in algorithmic creativity comparable to temperature sampling. For the larger models (Fig 5), seed-conditioning generally outperforms the baseline "null" conditioning (which appears mode-collapsed, see Fig 18, 19, 20). For the smaller model as well (Fig 6), we find gains in all tasks except for Sibling Discovery. Besides, for any fixed temperature, prefixing a seed only improves performance over a null prefix (Fig 6, 18). In §I.2, we find that the improved creativity from seedconditioning comes from improved diversity, rather than from reduced memorization (unlike with multi-token training). All these behaviors of seed-conditioning require further study as they are non-trivial, especially given that the seeds are naively, arbitrarily chosen in relationship to the output. As a starting point, we provide some preliminary

Temperature for sampling (trained with NTP)
greedy temp0.5 temp1.0 temp2.0 Sibling Discovery Triangle Discovery seed null 0.00 0.25 0.50 0.75 1.00 seed null 0.000 0.025 0.050 0.075 0.100 Cr ea ti vi ty Cr ea ti vi ty Circle Construction Line Construction Cre at iv it y seed null 0.0 0.2 0.4 Cre at iv it y seed null 0.0 0.2 0.4
hypotheses for when and why seed-conditioning may help Transformers in §C.1. Note that we do not see gains of seedconditioning when it comes to diffusion training (§ G.3). Robustness to hyperparameters. In §F and Fig 22, we do sensitivity analysis on all the datasets. We report how our above findings are robust to the choice of learning rate, batch-size, number of training steps, weight given to the multi-token objective, varying sampling conditions and reasonable changes to the complexity of the dataset and training set size (as per our argument in §2.6, we do expect the next-vs. multi-token gap to diminish for larger dataset size).

## 4.2. An Exploration Of Real-World Summarization

For a more realistic examination, we finetune GPT models NTP and the multi-token teacherless objectives on summarization tasks (XSUM, CNN/DailyMail). We measure the diversity of a model for any given prompt by generating 5 different completions and computing a Self-Bleu metric (Zhu et al., 2018). Admittedly though, a summarization task is not as open-ended as we would like: a higher quality model (i.e., higher Rouge; Lin, 2004) necessarily means lower diversity. To account for this, we plot how diversity evolves over time as a function of generation quality; we then find in Fig 7 that for a given model quality, the larger multi-token models achieve higher diversity (albeit only by a slight amount). This increase does not hold for smaller models and is not always noticeable for CNN/DailyMail (see
§J). Interestingly, teacherless training consistently shows an increase in summarization quality, measured by Rouge.

 Training Objective: Standard (Next-Token) Teacherless (Multi-Token) 
Div ers it y (
1 -
 Sel f-
BL
EU)XSUM-GPT-XL
Div ers it y ( 1 -
 Sel f-
BL
EU)XSUM-GPT-LARGE
0.135 0.140 0.145 0.150 0.155 0.160 Quality (ROUGE)
0.935 0.940 0.945 0.950 0.955 0.960 0.965 0.135 0.140 0.145 0.150 0.155 Quality (ROUGE)
0.935 0.940 0.945 0.950 0.955 0.960 0.965

## 5. Related Work

Open-ended algorithmic tasks. Directly related to us are Khona et al. (2024); Allen-Zhu & Li (2023b) who study diversity of next-token-trained models on an openended algorithmic task. Khona et al. (2024) demonstrate a diversity-accuracy tradeoff under temperature-scaling for path-connectivity on on a knowledge graph. We show how this tradeoff can be improved under alternative training methods. Allen-Zhu & Li (2023b) empirically demonstrate that next-token predictors are able to learn a synthetic, challenging CFG, in the "infinite" data regime (≈ 100m tokens).

Our datasets are not CFGs, with the exception of Sibling Discovery. Our negative result does not contradict theirs since what we show is a sub-optimality of NTP in a smaller data regime. Our work also extends the above works by studying limitations in much more minimal tasks that require as little as 2-hop lookahead. There are other works that study Transformers on non-open-ended graph-algorithmic tasks, discussed in §K. Diversity in generative models. Generative diversity has long been a major goal, at least until the revolution in (deterministic) reasoning of language models. Much work has gone into concerns such as mode collapse (Che et al., 2017) or posterior collapse (Bowman et al., 2016) and memorization (Carlini et al., 2020; 2023; Nasr et al., 2023). Temperature sampling has been known to have weak correlation with creativity, often inadvertently introducing incoherence (Peeperkorn et al., 2024; Chen & Ding, 2023; Chung et al., 2023). Our results on seed-conditioning are also reminiscent of a line of work on reinforcement learning (RL) showing that adding noises to the policy model parameters enables more efficient exploration than directly adding noises to the output space (Plappert et al., 2017; Fortunato et al., 2017). We defer discussion of theoretical studies of diversity and memorization and with empirical studies on natural-language creativity in §K.

Going beyond next-token prediction (NTP). There has been a recent emerging discussion surrounding the role of NTP as foundational piece in developing intelligent models. On the critical side, arguments have been made about the inference-time issues with auto-regression (Dziri et al., 2024; LeCun, 2024; Kääriäinen, 2006; Ross & Bagnell, 2010). Others have reported the planning and arithmetic limitations of next-token trained models (McCoy et al., 2023; Momennejad et al., 2023; Valmeekam et al., 2023a;b;c; Bachmann & Nagarajan, 2024) where the goal is accuracy, not diversity. As for diffusion as an alternative to NTP, our findings parallel that of Ye et al. (2024) who show that their variant of diffusion is able to solve the challenging path-star task of B&N'24. We provide references to more lines of multi-token prediction work in §K. There are also other Transformer failures such as the reversal curse (Allen- Zhu & Li, 2023a) or shortcut-learning (Dziri et al., 2024; Zhang et al., 2023; Liu et al., 2023; Young & You, 2022; Lai et al., 2021; Ranaldi & Zanzotto, 2023), however these are out-of-distribution failures; the sub-optimality we show is in-distribution, like in B&N'24. Injecting noise into a Transformer. Most related to seedconditioning is DeSalvo et al. (2024) who induce diversity by varying a *soft*-prompt learned using a reconstruction loss.

Our approach requires no modification to the architecture or the loss; however, we train the whole model, which is more expensive. A concurrent position paper (Jahrens &
Martinetz, 2025) conceptually suggests injecting noise with the same motivation as us. Seed-conditioning may also be viewed as a controllable way to vary prompt wordings which is known to induce diverse outputs (Li et al., 2023; Lau et al., 2024; Naik et al., 2024; Li et al., 2022).

## 6. Conclusions

This work provides a minimal test-bed of tasks abstracting distinct modes of creativity. While these tasks are admittedly an extreme caricaturization of real-world tasks, they enable us to quantify otherwise elusive metrics like originality and diversity. They also enable us to control and investigate distinct parts of the current apparatus for language modeling (next-token learning and temperature sampling) and advocate for alternatives (multi-token learning and seed-conditioning). Next, one can investigate the effect of length generalization, scaling and the effect of in-context learning vs. finetuning. The surprising effectiveness of seed-conditioning raises various open questions (§C.3). Other profound questions arise, such as whether reasoning-enhancing methods like RL and CoT are optimal for enhancing open-ended diversity and originality (§C.2).

We hope our work inspires discussion in the various directions of multi-token prediction, creativity and planning. We present more discussions in §C and limitations in §A.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning through the study of simple algorithmic tasks inspired by creativity. There are many potential societal consequences of our work - especially if one applies AI to real-world creative endeavors - none which we feel must be specifically highlighted in our focused algorithmic study.

## Acknowledgements

We wish to thank Gregor Bachmann, Jacob Springer, and Sachin Goyal for extensive feedback on a draft of the paper. We are grateful to Vansh Bansal whose significant effort in an independent reproduction of our GPT-2 (86M) experiments helped zero in on the role of top-K sampling in seed-conditioning for some of our datasets which we had overlooked in initial versions of the paper. We wish to thank Quentin Fournier for excellent references to work on AI for scientific discovery. We thank Garret Tanzer and Elan Rosenfeld for helping arrive at the term "seed-conditioning". We also wish to thank Mike Mozer, Suhas Kotha, Clayton Sanford, Christina Baek, Yuxiao Qu, Ziqian Zhong for valuable discussions and pointers. The work was supported in part by Cisco, Apple, Google, OpenAI, NSF, Okawa foundation and Schmidt Sciences.

## References

Alabdulmohsin, I., Tran, V. Q., and Dehghani, M. Fractal patterns may unravel the intelligence in next-token prediction, 2024.

Allen-Zhu, Z. and Li, Y. Physics of language models: Part 3.2, knowledge manipulation. *CoRR*, abs/2309.14402, 2023a. doi: 10.48550/ARXIV.2309.14402. URL https: //doi.org/10.48550/arXiv.2309.14402.

Allen-Zhu, Z. and Li, Y. Physics of language models: Part 1, context-free grammar. *CoRR*, abs/2305.13673, 2023b.

doi: 10.48550/ARXIV.2305.13673. URL https://doi. org/10.48550/arXiv.2305.13673.

Anderson, B. R., Shah, J. H., and Kreminski, M. Homogenization effects of large language models on human creative ideation. In Proceedings of the 16th Conference on Creativity & Cognition, Chicago, IL, USA, June 23-26, 2024, pp. 413–425. ACM, 2024. URL https://doi.org/10.1145/3635636.3656204.

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and van den Berg, R. Structured denoising diffusion models in discrete state-spaces. *NeurIPS*, 2021.

Bachmann, G. and Nagarajan, V. The pitfalls of next-token prediction. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 2296–2318, 2024.

Bavarian, M., Jun, H., Tezak, N. A., Schulman, J.,
McLeavey, C., Tworek, J., and Chen, M. Efficient training of language models to fill in the middle.

ArXiv, abs/2207.14255, 2022. URL https://api.

semanticscholar.org/CorpusID:251135268.

Beel, J., Kan, M.-Y., and Baumgart, M. Evaluating sakana's ai scientist for autonomous research: Wishful thinking or an emerging reality towards 'artificial research intelligence' (ari)?, 2025.

Berglund, L., Tong, M., Kaufmann, M., Balesni, M., Stickland, A. C., Korbak, T., and Evans, O. The reversal curse: Llms trained on "a is b" fail to learn "b is a". In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024.

OpenReview.net, 2024. URL https://openreview. net/forum?id=GPKTIktA0k.

Bigelow, E. J., Lubana, E. S., Dick, R. P., Tanaka, H., and Ullman, T. D. In-context learning dynamics with random binary sequences. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

Boden, M. A. The Creative Mind - Myths and Mechanisms
(2. ed.). Routledge, 2003.

Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Józefowicz, R., and Bengio, S. Generating sentences from a continuous space. In *Proceedings of the 20th SIGNLL* Conference on Computational Natural Language Learning, CoNLL 2016,, pp. 10–21. ACL, 2016.

Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J.,
Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S., et al. Sparks of artificial general intelligence: Early experiments with gpt-4. *arXiv preprint* arXiv:2303.12712, 2023.

Callaway, E. Cognitive science: Leap of thought. *Nature*,
502, 2013.

Carlini, N., Tramèr, F., Wallace, E., Jagielski, M., Herbert-
Voss, A., Lee, K., Roberts, A., Brown, T. B., Song, D. X., Erlingsson, Ú., Oprea, A., and Raffel, C. Extracting training data from large language models. In *USENIX*
Security Symposium, 2020.

Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramèr, F.,
and Zhang, C. Quantifying memorization across neural language models. *ICLR*, 2023.

Chakrabarty, T., Laban, P., Agarwal, D., Muresan, S., and Wu, C. Art or artifice? large language models and the false promise of creativity. In *Proceedings of the CHI* Conference on Human Factors in Computing Systems, CHI 2024, Honolulu, HI, USA, May 11-16, 2024, pp. 30:1–30:34. ACM, 2024.

Che, T., Li, Y., Jacob, A. P., Bengio, Y., and Li, W. Mode regularized generative adversarial networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017.

Chen, H. and Ding, N. Probing the "creativity" of large language models: Can models produce divergent semantic association? In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pp. 12881–12888. Association for Computational Linguistics, 2023.

Chow, Y., Tennenholtz, G., Gur, I., Zhuang, V., Dai, B., Thiagarajan, S., Boutilier, C., Agarwal, R., Kumar, A., and Faust, A. Inference-aware fine-tuning for best-of-n sampling in large language models. *ArXiv*, abs/2412.15287, 2024. URL https://api.semanticscholar.org/ CorpusID:274965054.

Chung, J. J. Y., Kamar, E., and Amershi, S. Increasing diversity while maintaining accuracy: Text data generation with large language models and human interventions.

In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL, pp. 575–593. Association for Computational Linguistics, 2023.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,
Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.

Csikszentmihalyi, M. Creativity: Flow and the Psychology of Discovery and Invention. HarperCollins Publishers, New York, NY, first edition, 1996.

Dang, X., Baek, C., Wen, K., Kolter, Z., and Raghunathan, A. Weight ensembling improves reasoning in language models. 2025. URL https://api.semanticscholar.

org/CorpusID:277781120.

Dawid, A. and LeCun, Y. Introduction to latent variable energy-based models: A path towards autonomous machine intelligence. *arXiv preprint arXiv:2306.02572*, 2023.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *ArXiv*,
abs/2501.12948, 2025.

DeepSeek-AI, Liu, A., Feng, B., Xue, B., Wang, B.-L., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Guo, D., Yang, D., Chen, D., Ji, D.-L., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Zhang, H., Ding, H., Xin, H., Gao, H., Li, H., Qu, H., Cai, J. L., Liang, J., Guo, J., Ni, J., Li, J., Wang, J., Chen, J., Chen, J., Yuan, J., Qiu, J., Li, J., Song, J.-M., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Xu, L., Xia, L., Zhao, L., Wang, L., Zhang, L., Li, M., Wang, M., Zhang, M., Zhang, M., Tang, M., Li, M., Tian, N., Huang, P., Wang, P., Zhang, P., Wang, Q., Zhu, Q., Chen, Q., Du, Q., Chen, R. J., Jin, R. L., Ge, R., Zhang, R., Pan, R., Wang, R., Xu, R., Zhang, R., Chen, R., Li, S. S., Lu, S., Zhou, S., Chen, S., Wu, S.-P., Ye, S., Ma, S., Wang, S., Zhou, S., Yu, S., Zhou, S., Pan, S., Wang, T., Yun, T., Pei, T., Sun, T., Xiao, W. L., Zeng, W., Zhao, W., An, W., Liu, W., Liang, W., Gao, W., Yu, W.-X., Zhang, W., Li, X. Q., Jin, X., Wang, X., Bi, X., Liu, X., Wang, X., Shen, X.-C.,
Chen, X., Zhang, X., Chen, X., Nie, X., Sun, X., Wang, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yu, X., Song, X., Shan, X., Zhou, X., Yang, X., Li, X., Su, X., Lin, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhu, Y. X., Zhang, Y., Xu, Y., Huang, Y., Li, Y., Zhao, Y., Sun, Y., Li, Y., Wang, Y., Yu, Y., Zheng, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Tang, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y.-B., Liu, Y., Guo, Y., Wu, Y., Ou, Y., Zhu, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Zha, Y., Xiong, Y., Ma, Y., Yan, Y., Luo, Y.-W., mei You, Y., Liu, Y., Zhou, Y., Wu, Z. F., Ren, Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Huang, Z., Zhang, Z., Xie, Z., guo Zhang, Z., Hao, Z., Gou, Z., Ma, Z., Yan, Z., Shao, Z., Xu, Z., Wu, Z., Zhang, Z., Li, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z.-A., Xie, Z., Song, Z., Gao, Z., and Pan, Z. Deepseek-v3 technical report. In *ArXiv*, 2024.

DeSalvo, G., Kagy, J.-F., Karydas, L., Rostamizadeh, A.,
and Kumar, S. No more hard prompts: Softsrv prompting for synthetic data generation, 2024. URL https: //arxiv.org/abs/2410.16534.

Du, L., Mei, H., and Eisner, J. Autoregressive modeling with lookahead attention. *arXiv preprint arXiv:2305.12272*,
2023.

Dziri, N., Lu, X., Sclar, M., Li, X. L., Jiang, L., Lin, B. Y.,
Welleck, S., West, P., Bhagavatula, C., Le Bras, R., et al. Faith and fate: Limits of transformers on compositionality.

Advances in Neural Information Processing Systems, 36, 2024.

Feng, G., Zhang, B., Gu, Y., Ye, H., He, D., and Wang, L.

Towards revealing the mystery behind chain of thought: a theoretical perspective. *Advances in Neural Information* Processing Systems, 36, 2023.

Finlayson, M., Hewitt, J., Koller, A., Swayamdipta, S., and Sabharwal, A. Closing the curious case of neural text degeneration. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

Fortunato, M., Azar, M. G., Piot, B., Menick, J., Osband, I.,
Graves, A., Mnih, V., Munos, R., Hassabis, D., Pietquin, O., Blundell, C., and Legg, S. Noisy networks for exploration. *ArXiv*, abs/1706.10295, 2017. URL https: //api.semanticscholar.org/CorpusID:5176587.

Franceschelli, G. and Musolesi, M. On the creativity of large language models. *CoRR*, abs/2304.00008, 2023.

Fried, D., Aghajanyan, A., Lin, J., Wang, S. I., Wallace, E., Shi, F., Zhong, R., tau Yih, W., Zettlemoyer, L., and Lewis, M. Incoder: A generative model for code infilling and synthesis. *ArXiv*, abs/2204.05999, 2022.

Frydenlund, A. The mystery of the pathological path-star task for language models. In *Proceedings of the 2024* Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pp. 12493–12516. Association for Computational Linguistics, 2024.

Gemma Team, T. M., Hardin, C., Dadashi, R., Bhupatiraju, S., Sifre, L., Rivière, M., Kale, M. S., Love, J., Tafti, P.,
Hussenot, L., and et al. Gemma. 2024. doi: 10.34740/
KAGGLE/M/3301. URL https://www.kaggle.com/ m/3301.

Gloeckle, F., Idrissi, B. Y., Rozière, B., Lopez-Paz, D., and Synnaeve, G. Better & faster large language models via multi-token prediction. 2024.

Gong, S., Li, M., Feng, J., Wu, Z., and Kong, L. Diffuseq:
Sequence to sequence text generation with diffusion models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B.,
Warde-Farley, D., Ozair, S., Courville, A. C., and Bengio, Y. Generative adversarial networks. *Commun. ACM*, 63(11):139–144, 2020. doi: 10.1145/3422622. URL https://doi.org/10.1145/3422622.

Goyal, A., Sordoni, A., Côté, M.-A., Ke, N. R., and Bengio, Y. Z-forcing: Training stochastic recurrent networks. NeurIPS, 2017.

Goyal, S., Ji, Z., Rawat, A. S., Menon, A. K., Kumar, S., and Nagarajan, V. Think before you speak: Training language models with pause tokens. The Twelfth International Conference on Learning Representations, ICLR 2024, 2024.

Gruver, N., Stanton, S., Frey, N. C., Rudner, T. G. J., Hotzel, I., Lafrance-Vanasse, J., Rajpal, A., Cho, K., and Wilson, A. G. Protein design with guided discrete diffusion. *ArXiv*, abs/2305.20009, 2023. URL https://api. semanticscholar.org/CorpusID:258987335.

Gu, J., Bradbury, J., Xiong, C., Li, V. O. K., and Socher, R. Non-autoregressive neural machine translation. In 6th International Conference on Learning Representations, ICLR 2018, Conference Track Proceedings. Open- Review.net, 2018.

Gupta, T. and Pruthi, D. All that glitters is not novel: Plagiarism in ai generated research, 2025. URL https:
//arxiv.org/abs/2502.16487.

Hayes, T., Rao, R., Akin, H., Sofroniew, N. J., Oktay, D., Lin, Z., Verkuil, R., Tran, V. Q., Deaton, J., Wiggert, M., Badkundri, R., Shafkat, I., Gong, J., Derry, A., Molina, R. S., Thomas, N., Khan, Y. A., Mishra, C., Kim, C., Bartie, L. J., Nemeth, M., Hsu, P. D., Sercu, T., Candido, S., and Rives, A. Simulating 500 million years of evolution with a language model. *Science*, 387(6736):850–858, 2025. doi: 10.1126/ science.ads0018. URL https://www.science.org/ doi/abs/10.1126/science.ads0018.

Hofstadter, D. A review of mental leaps: Analogy in creative thought. *AI Mag.*, 16(3):75–80, 1995. doi: 10. 1609/AIMAG.V16I3.1154. URL https://doi.org/ 10.1609/aimag.v16i3.1154.

Holyoak, K. J. and Thagard, P. Mental leaps: analogy in creative thought. MIT Press, Cambridge, MA, USA, 1995. ISBN 0262082330.

Hoogeboom, E., Nielsen, D., Jaini, P., Forr'e, P., and Welling, M. Argmax flows and multinomial diffusion: Learning categorical distributions. In Neural Information Processing Systems, 2021.

Hopkins, A. K., Renda, A., and Carbin, M. Can LLMs generate random numbers? evaluating LLM sampling in controlled domains. In ICML 2023 Workshop: Sampling and Optimization in Discrete Space, 2023. URL https: //openreview.net/forum?id=Vhh1K9LjVI.

Hu, E. S., Ahn, K., Liu, Q., Xu, H., Tomar, M., Langford, A., Jayaraman, D., Lamb, A., and Langford, J. The belief state transformer. In *The Thirteenth International* Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

Hua, H., Li, X., Dou, D., Xu, C., and Luo, J. Fine-tuning pre-trained language models with noise stability regularization. *CoRR*, 2022.

Jahrens, M. and Martinetz, T. Why llms cannot think and how to fix it, 2025. URL https://arxiv.org/abs/ 2503.09211.

Jain, N., Chiang, P., Wen, Y., Kirchenbauer, J., Chu, H., Somepalli, G., Bartoldson, B. R., Kailkhura, B., Schwarzschild, A., Saha, A., Goldblum, M., Geiping, J.,
and Goldstein, T. Neftune: Noisy embeddings improve instruction finetuning. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

Jansen, P. A., Cot'e, M.-A., Khot, T., Bransom, E., Dalvi, B., Majumder, B. P., Tafjord, O., and Clark, P. Discoveryworld: A virtual environment for developing and evaluating automated scientific discovery agents. *NeurIPS*,
2024.

Kääriäinen, M. Lower bounds for reductions. In *Atomic* Learning Workshop, 2006.

Kalai, A. T. and Vempala, S. S. Calibrated language models must hallucinate. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, STOC 2024, Vancouver, BC, Canada, June 24-28, 2024, pp. 160–171. ACM, 2024.

Kalavasis, A., Mehrotra, A., and Velegkas, G. On the limits of language generation: Trade-offs between hallucination and mode collapse. abs/2411.09642, 2024.

Kamb, M. and Ganguli, S. An analytic theory of creativity in convolutional diffusion models, 2024. URL https: //arxiv.org/abs/2412.20292.

Khona, M., Okawa, M., Hula, J., Ramesh, R., Nishi, K.,
Dick, R. P., Lubana, E. S., and Tanaka, H. Towards an understanding of stepwise inference in transformers: A synthetic graph navigation model. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. In 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014. URL http://arxiv.org/abs/1312.6114.

Kitouni, O., Nolte, N., Bouchacourt, D., Williams, A.,
Rabbat, M., and Ibrahim, M. The factorization curse: Which tokens you predict underlie the reversal curse and more. *CoRR*, abs/2406.05183, 2024. doi: 10.

48550/ARXIV.2406.05183. URL https://doi.org/ 10.48550/arXiv.2406.05183.

Kleinberg, J. M. and Mullainathan, S. Language generation in the limit. CoRR, abs/2404.06757, 2024. doi: 10. 48550/ARXIV.2404.06757. URL https://doi.org/
10.48550/arXiv.2404.06757.

Lai, Y., Zhang, C., Feng, Y., Huang, Q., and Zhao, D. Why machine reading comprehension models learn shortcuts? In Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021, volume ACL/IJCNLP 2021 of *Findings of ACL*, pp. 989–1002. Association for Computational Linguistics, 2021.

Lau, G. K. R., Hu, W., Liu, D., Chen, J., Ng, S.-K., and Low, B. K. H. Dipper: Diversity in prompts for producing large language model ensembles in reasoning tasks, 2024.

URL https://arxiv.org/abs/2412.15238.

LeCun, Y. Do large language models need sensory grounding for meaning and understanding? University Lecture, 2024.

Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., and Papailiopoulos, D. Teaching arithmetic to small transformers. In *The Twelfth International Conference on Learning* Representations, ICLR 2024. OpenReview.net, 2024.

Li, Y., Choi, D. H., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Lago, A. D., Hubert, T., Choy, P., de Masson d'Autume, C., Babuschkin, I., Chen, X., Huang, P., Welbl, J., Gowal, S., Cherepanov, A., Molloy, J., Mankowitz, D. J., Robson, E. S., Kohli, P., de Freitas, N., Kavukcuoglu, K., and Vinyals, O. Competition-level code generation with alphacode. 2022.

Li, Y., Lin, Z., Zhang, S., Fu, Q., Chen, B., Lou, J.-G., and Chen, W. Making language models better reasoners with step-aware verifier. In *Proceedings of the 61st Annual* Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Toronto, Canada, July 2023.

Association for Computational Linguistics. URL https: //aclanthology.org/2023.acl-long.291/.

Lin, C.-Y. ROUGE: A package for automatic evaluation of summaries. In *Text Summarization Branches Out*, pp.

74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.

org/W04-1013/.

Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A., and Zhang, C. Transformers learn shortcuts to automata. In The Eleventh International Conference on Learning Representations, ICLR 2023, 2023.

Lou, A., Meng, C., and Ermon, S. Discrete diffusion modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, 2023.

Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., and Ha, D. The ai scientist: Towards fully automated open-ended scientific discovery, 2024a.

Lu, X., Sclar, M., Hallinan, S., Mireshghallah, N., Liu, J.,
Han, S., Ettinger, A., Jiang, L., Chandu, K. R., Dziri, N., and Choi, Y. AI as humanity's salieri: Quantifying linguistic creativity of language models via systematic attribution of machine text against web text. abs/2410.04265, 2024b.

Madani, A., Krause, B., Greene, E. R., Subramanian, S.,
Mohr, B. P., Holton, J. M., Olmos, Jr., J. L., Xiong, C., Sun, Z. Z., Socher, R., Fraser, J. S., and Naik, N. Large language models generate functional protein sequences across diverse families. *Nature Biotechnology*, 41(8), 1 2023. doi: 10.1038/s41587-022-01618-2.

Malach, E. Auto-regressive next-token predictors are universal learners. *arXiv preprint arXiv:2309.06979*, 2023.

McCoy, R. T., Yao, S., Friedman, D., Hardy, M., and Griffiths, T. L. Embers of autoregression: Understanding large language models through the problem they are trained to solve. *arXiv preprint arXiv:2309.13638*, 2023.

McLaughlin, A., Campbell, J., Uppuluri, A., and Yang, Y.

Aidanbench: Stress-testing language model creativity on open-ended questions. In *NeurIPS 2024 Workshop on* Language Gamification, 2024.

Meier, J., Rao, R., Verkuil, R., Liu, J., Sercu, T., and Rives, A. Language models enable zero-shot prediction of the effects of mutations on protein function. In Advances in Neural Information Processing Systems, volume 34, pp. 29287–29303, 2021.

Merrill, W. and Sabharwal, A. The expressive power of transformers with chain of thought. In The Twelfth International Conference on Learning Representations, 2024.

Mirowski, P. W., Love, J., Mathewson, K. W., and Mohamed, S. A robot walks into a bar: Can language models serve as creativity support tools for comedy? an evaluation of llms' humour alignment with comedians. *CoRR*,
abs/2405.20956, 2024.

Momennejad, I., Hasanbeig, H., Frujeri, F. V., Sharma, H.,
Ness, R. O., Jojic, N., Palangi, H., and Larson, J. Evaluating cognitive maps and planning in large language models with cogeval. *Advances in Neural Information* Processing Systems, 36, 2023.

Monea, G., Joulin, A., and Grave, E. Pass: Parallel speculative sampling. 3rd Workshop on Efficient Natural Language and Speech Processing (NeurIPS 2023), 2023.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Candès, E. J., and Hashimoto, T. s1: Simple test-time scaling.

abs/2501.19393, 2025.

Nagarajan, V., Raffel, C., and Goodfellow, I. J. Theoretical insights into memorization in gans. In Neural Information Processing Systems Workshop, volume 1, pp. 3, 2018.

Naik, R., Chandrasekaran, V., Yuksekgonul, M., Palangi, H.,
and Nushi, B. Diversity of thought improves reasoning abilities of llms, 2024. URL https://arxiv.org/abs/ 2310.07088.

Nakkiran, P., Bradley, A., Zhou, H., and Advani, M. Stepby-step diffusion: An elementary tutorial, 2024.

Nallapati, R., Zhou, B., dos santos, C. N., Gulcehre, C., and Xiang, B. Abstractive text summarization using sequenceto-sequence rnns and beyond, 2016. URL https:// arxiv.org/abs/1602.06023.

Narayan, S., Cohen, S. B., and Lapata, M. Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization, 2018. URL
https://arxiv.org/abs/1808.08745.

Nasr, M., Carlini, N., Hayase, J., Jagielski, M., Cooper, A. F., Ippolito, D., Choquette-Choo, C. A., Wallace, E., Tramèr, F., and Lee, K. Scalable extraction of training data from (production) language models. *ArXiv*, 2023.

Nguyen, E., Poli, M., Durrant, M. G., Kang, B., Katrekar, D., Li, D. B., Bartie, L. J., Thomas, A. W., King, S. H., Brixi, G., Sullivan, J., Ng, M. Y., Lewis, A., Lou, A., Ermon, S., Baccus, S. A., Hernandez-Boussard, T., Ré, C., Hsu, P. D., and Hie, B. L. Sequence modeling and design from molecular to genome scale with evo. *Science*, 386(6723):eado9336, 2024. doi: 10.1126/ science.ado9336. URL https://www.science.org/ doi/abs/10.1126/science.ado9336.

Nolte, N., Kitouni, O., Williams, A., Rabbat, M., and Ibrahim, M. Transformers can navigate mazes with multi-step prediction. *CoRR*, abs/2412.05117, 2024. doi: 10.48550/ARXIV.2412.05117. URL https://doi. org/10.48550/arXiv.2412.05117.

Okawa, M., Lubana, E. S., Dick, R. P., and Tanaka, H. Compositional abilities emerge multiplicatively: Exploring diffusion models on a synthetic task. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

OpenAI. Openai o1 system card. *ArXiv*, 2024. Padmakumar, V. and He, H. Does writing with language models reduce content diversity? In The Twelfth International Conference on Learning Representations, ICLR
2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=
Feiz5HtCD0.

Pannatier, A., Courdier, E., and Fleuret, F. σ-gpts: A new approach to autoregressive models. In Machine Learning and Knowledge Discovery in Databases. Research Track - European Conference, ECML PKDD 2024, Vilnius, Lithuania, September 9-13, 2024, Proceedings, Part VII, volume 14947 of *Lecture Notes in Computer Science*, pp. 143–159. Springer, 2024.

Peeperkorn, M., Kouwenhoven, T., Brown, D., and Jordanous, A. Is temperature the creativity parameter of large language models? abs/2405.00492, 2024.

Pezeshki, M., Kaba, S., Bengio, Y., Courville, A. C., Precup, D., and Lajoie, G. Gradient starvation: A learning proclivity in neural networks. In *Advances in Neural Information* Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 1256–1272, 2021.

Plappert, M., Houthooft, R., Dhariwal, P., Sidor, S.,
Chen, R. Y., Chen, X., Asfour, T., Abbeel, P., and Andrychowicz, M. Parameter space noise for exploration. *ArXiv*, abs/1706.01905, 2017. URL https: //api.semanticscholar.org/CorpusID:2971655.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multitask learners. 2019.

Ranaldi, L. and Zanzotto, F. M. Hans, are you clever? clever hans effect analysis of neural systems, 2023.

Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J.,
Guo, D., Ott, M., Zitnick, C. L., Ma, J., and Fergus, R. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proc. Natl. Acad. Sci. USA, 118(15):e2016239118, 2021.

Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J. R., Ellenberg, J. S., Wang, P., Fawzi, O., Kohli, P., and Fawzi, A. Mathematical discoveries from program search with large language models. *Nat.*, 625(7995), 2024.

Ross, S. and Bagnell, D. Efficient reductions for imitation learning. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, AIS- TATS 2010, Chia Laguna Resort, Sardinia, Italy, May 13-15, 2010, volume 9 of *JMLR Proceedings*, 2010.

Runco, M. A. and Jaeger, G. J. The standard definition of creativity. *Creativity Research Journal*, 24(1):92–96, 2012.

Sanford, C., Fatemi, B., Hall, E., Tsitsulin, A., Kazemi, S. M., Halcrow, J., Perozzi, B., and Mirrokni, V. Understanding transformer reasoning capabilities via graph algorithms. abs/2405.18512, 2024.

Saparov, A., Pawar, S., Pimpalgaonkar, S., Joshi, N., Pang, R. Y., Padmakumar, V., Kazemi, S. M., Kim, N., and He, H. Transformers struggle to learn to search, 2024. URL
https://arxiv.org/abs/2412.04703.

Schmidhuber, J. Driven by compression progress: A simple principle explains essential aspects of subjective beauty, novelty, surprise, interestingness, attention, curiosity, creativity, art, science, music, jokes. In Computational Creativity: An Interdisciplinary Approach, 12.07. - 17.07.2009, Dagstuhl Seminar Proceedings, 2009.

Schnitzler, J., Ho, X., Huang, J., Boudin, F., Sugawara, S., and Aizawa, A. Morehopqa: More than multi-hop reasoning. abs/2406.13397. doi: 10.48550/ARXIV.2406.

13397.

Shah, H., Tamuly, K., Raghunathan, A., Jain, P., and Netrapalli, P. The pitfalls of simplicity bias in neural networks. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https: //proceedings.neurips.cc/paper/2020/hash/ 6cfe0e6127fa25df2a0ef2ae1067d915-Abstract. html.

Shannon, C. E. A mathematical theory of communication.

The Bell System Technical Journal, 27(3):379–423, 1948.

Shannon, C. E. Prediction and entropy of printed english.

The Bell System Technical Journal, 30(1):50–64, 1951.

Shlegeris, B., Roger, F., Chan, L., and McLean, E. Language models are better than humans at next-token prediction. arXiv preprint arXiv:2212.11281, 2022.

Si, C., Yang, D., and Hashimoto, T. Can llms generate novel research ideas? A large-scale human study with 100+ NLP researchers. 2024.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm test-time compute optimally can be more effective than scaling model parameters. *ArXiv*, abs/2408.03314, 2024. URL https://api.semanticscholar.org/ CorpusID:271719990.

Talmor, A., Tafjord, O., Clark, P., Goldberg, Y., and Berant, J. Leap-of-thought: Teaching pre-trained models to systematically reason over implicit knowledge. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

Thankaraj, A., Jiang, Y., Kolter, J. Z., and Bisk, Y. Looking beyond the next token, 2025. URL https://arxiv.

org/abs/2504.11336.

Tschannen, M., Kumar, M., Steiner, A., Zhai, X., Houlsby, N., and Beyer, L. Image captioners are scalable vision learners too. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pp. 13484– 13508. Association for Computational Linguistics, 2023.

Wang, Y., Luo, X., Wei, F., Liu, Y., Zhu, Q., Zhang, X.,
Yang, Q., Xu, D., and Che, W. Make some noise: Unlocking language model parallel inference capability through noisy training. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pp. 12914–12926. Association for Computational Linguistics, 2024c.

Valmeekam, K., Marquez, M., and Kambhampati, S. Can large language models really improve by self-critiquing their own plans? *arXiv preprint arXiv:2310.08118*,
2023a.

Valmeekam, K., Marquez, M., Olmo, A., Sreedharan, S.,
and Kambhampati, S. Planbench: An extensible benchmark for evaluating large language models on planning and reasoning about change, 2023b.

Watson, J., Juergens, D., Bennett, N., Trippe, B., Yim, J.,
Eisenach, H., Ahern, W., Borst, A., Ragotte, R., Milles, L., Wicky, B., Hanikel, N., Pellock, S., Courbet, A., Sheffler, W., Wang, J., Venkatesh, P., Sappington, I., Torres, S., Lauko, A., {De Bortoli}, V., Mathieu, E., Ovchinnikov, S., Barzilay, R., Jaakkola, T., DiMaio, F., Baek, M., and Baker, D. De novo design of protein structure and function with rfdiffusion. *Nature*, 620:1089–1100, 2023.

Valmeekam, K., Marquez, M., Sreedharan, S., and Kambhampati, S. On the planning abilities of large language models - A critical investigation. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023c.

Varshney, L. R., Pinel, F., Varshney, K. R., Bhattacharjya, D., Schörgendorfer, A., and Chee, Y. A big data approach to computational creativity: The curious case of chef watson. *IBM J. Res. Dev.*, 63(1):7:1–7:18, 2019.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B.,
Xia, F., Chi, E. H., Le, Q. V., and Zhou, D. Chain-ofthought prompting elicits reasoning in large language models. In *Advances in Neural Information Processing* Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, 2022.

Walsh, M., Preus, A., and Gronski, E. Does chatgpt have a poetic style? In Proceedings of the Computational Humanities Research Conference 2024, Aarhus, Denmark, December 4-6, 2024, volume 3834 of CEUR Workshop Proceedings, pp. 1201–1219. CEUR-WS.org.

Wies, N., Levine, Y., and Shashua, A. Sub-task decomposition enables learning in sequence to sequence tasks.

In *The Eleventh International Conference on Learning* Representations, ICLR 2023, 2023.

Wang, H., Zhao, Y., Li, D., Wang, X., Liu, G., Lan, X., and Wang, H. Innovative thinking, infinite humor: Humor research of large language models through structured thought leaps. abs/2410.10370, 2024a.

Wu, Y., Sun, Z., Li, S., Welleck, S., and Yang, Y. Inference scaling laws: An empirical analysis of computeoptimal inference for problem-solving with language models. 2024. URL https://api.semanticscholar.

org/CorpusID:271601023.

Wang, H., Zou, J., Mozer, M., Goyal, A., Lamb, A., Zhang, L., Su, W. J., Deng, Z., Xie, M. Q., Brown, H., and Kawaguchi, K. Can ai be as creative as humans?, 2024b. URL https://arxiv.org/abs/2401.01623.

Xu, M., Jiang, G., Zhang, C., Zhu, S.-C., and Zhu, Y. Interactive visual reasoning under uncertainty. In *Neural* Information Processing Systems, 2022. URL https:// api.semanticscholar.org/CorpusID:249889691.

Wang, J. X., King, M., Porcel, N., Kurth-Nelson, Z., Zhu, T., Deck, C., Choy, P., Cassin, M., Reynolds, M., Song, F., Buttimore, G., Reichert, D. P., Rabinowitz, N. C., Matthey, L., Hassabis, D., Lerchner, A., and Botvinick, M. M. Alchemy: A benchmark and analysis toolkit for meta-reinforcement learning agents. In NeurIPS Datasets and Benchmarks, 2021. URL https://api. semanticscholar.org/CorpusID:239019925.

Yang, S., Gribovskaya, E., Kassner, N., Geva, M., and Riedel, S. Do large language models latently perform multi-hop reasoning? In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 10210–10229. Association for Computational Linguistics, 2024a.

Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A.,
Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning Yang, S., Kassner, N., Gribovskaya, E., Riedel, S., and Geva, M. Do large language models perform latent multi-hop reasoning without exploiting shortcuts? abs/2411.16679, 2024b. doi: 10.48550/ARXIV.2411.16679. URL https: //doi.org/10.48550/arXiv.2411.16679.

Yang, Z., Hu, Z., Salakhutdinov, R., and Berg-Kirkpatrick, T. Improved variational autoencoders for text modeling using dilated convolutions. *ICML*, 2017.

Yang, Z., Band, N., Li, S., Candès, E. J., and Hashimoto, T.

Synthetic continued pretraining. *CoRR*, abs/2409.07431, 2024c. doi: 10.48550/ARXIV.2409.07431. URL https: //doi.org/10.48550/arXiv.2409.07431.

Ye, J., Gao, J., Gong, S., Zheng, L., Jiang, X., Li, Z., and Kong, L. Beyond autoregression: Discrete diffusion for complex reasoning and planning. 2024. doi: 10.48550/ ARXIV.2410.14157.

Young, T. and You, Y. On the inconsistencies of conditionals learned by masked language models. arXiv preprint arXiv:2301.00068, 2022.

Yu, L., Jiang, W., Shi, H., YU, J., Liu, Z., Zhang, Y.,
Kwok, J., Li, Z., Weller, A., and Liu, W. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=N8N0hgNDRt.

Zhang, H., Li, L. H., Meng, T., Chang, K., and den Broeck, G. V. On the paradox of learning to reason from data. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI 2023, 19th25th August 2023, Macao, SAR, China, pp. 3365–3373. ijcai.org, 2023.

Zhang, J., Jain, L., Guo, Y., Chen, J., Zhou, K. L., Suresh, S.,
Wagenmaker, A., Sievert, S., Rogers, T. T., Jamieson, K., Mankoff, R., and Nowak, R. Humor in AI: massive scale crowd-sourced preferences and benchmarks for cartoon captioning. *CoRR*, abs/2406.10522, 2024a.

Zhang, Y., Schwarzschild, A., Carlini, N., Kolter, Z., and Ippolito, D. Forcing diffuse distributions out of language models. abs/2404.10859, 2024b.

Zhang, Y., Diddee, H., Holm, S., Liu, H., Liu, X.,
Samuel, V., Wang, B., and Ippolito, D. Noveltybench: Evaluating language models for humanlike diversity. 2025. URL https://api.semanticscholar.org/ CorpusID:277621515.

Zhao, Y., Zhang, R., Li, W., Huang, D., Guo, J., Peng, S., Hao, Y., Wen, Y., Hu, X., Du, Z., Guo, Q., Li, L., and Chen, Y. Assessing and understanding creativity in large language models. *ArXiv*, abs/2401.12491, 2024. URL https://api.semanticscholar.org/ CorpusID:267094860.

Zhong, S., Huang, Z., Gao, S., Wen, W., Lin, L., Zitnik, M.,
and Zhou, P. Let's think outside the box: Exploring leapof-thought in large language models with creative humor generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, 2024.

Zhu, Y., Lu, S., Zheng, L., Guo, J., Zhang, W., Wang, J.,
and Yu, Y. Texygen: A benchmarking platform for text generation models. The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, 2018.

## A. Limitations

We enumerate in detail the limitations of our work in terms of our experimental conclusions and in terms of our general approach to an abstraction of creativity.

## A.1. Limitations Of Our Experimental Conclusions

1. There may be many ways to improve upon next-token prediction for a minimal task. Unfortunately, success here does not necessarily guarantee success on more complex tasks. Conversely, minimal tasks are more valuable as a failure base case: failure here guarantees failure in more complex tasks.

2. Our examples do not preclude the existence of tasks where next-token prediction will outperform multi-token prediction; multi-token prediction is simply a more general-purpose objective suitable to lookahead tasks.

3. The teacherless multi-token prediction technique we explore as an alternative is generally harder to optimize than next-token prediction, especially for smaller models.

4. Even if multi-token approaches outperform next-token prediction relatively, in some of our simple tasks, all algorithms are far from delivering a sufficiently diverse model.

5. Although our tasks are minimal, we note that there is a certain range of hyperparameters (e.g., high degree or edge count) beyond which the models can struggle to learn them. We find that Triangle Discovery in particular is a challenging task, especially for smaller models. We also note that the models are curiously sensitive to the way the edges are formatted (see §G.5).

6. While seed-conditioning appears to be a competitive alternative to temperature sampling, there are caveats here:
(a) In our GPT-2 (86M) experiments for Sibling Discovery and Triangle Discovery, the gap between seedconditioning and temperature-sampling is contingent on top-K sampling (which is turned on by default in the library as it is standard practice in real-world applications). Although this does not seem to be a requirement in our other settings (and is immaterial for the other datasets which have a smaller vocabulary), it is not clear how crucial this factor would be for demonstrating the same gap in real settings.

(b) It is not clear how well this idea may generalize to realistic data.

(c) Unfortunately, seed-conditioning requires special training of the model, which is not as appealing as the far simpler temperature sampling.

7. There are many ways in which our observations have not been fully characterized or understood even in our minimal settings. For example, we do not thoroughly report on the effect of model size, model family, or pretraining; we do not analyze when and why seed-conditioning helps.

## A.2. Our Approach To Creativity

Below, we enumerate some important limitations of our approach towards building abstract and minimal models of creative tasks.

1. The skills we capture in our tasks are only (a subset of) computational skills *necessary* for creativity; these are far from being *sufficient*.

2. The type of algorithmic tasks we study capture only a tiny subset of creative tasks that fall under the taxonomy in Boden
(2003). There is yet another class called *transformative* creativity that we do not look at, and also other important taxonomies such as the Big-C/little-c creativity (Csikszentmihalyi, 1996). Big-C Creativity corresponds breakthroughs and world-changing ideas; what we focus on is adjacent to a class of little-c creativity tasks. Relatedly, many real-world creative tasks appear to be "out-of-distribution" in nature, which we do not capture.

3. Real-world creative tasks also apply over much larger context length and require drawing connections from a significantly larger memory (literally, the set of all things a human may know about). Our algorithmic tasks are tiny in comparison (although deliberately so).

4. Our empirical measure of creativity for algorithmic tasks is only a computationally-efficient proxy. Achieving an absolute high algorithmic creativity score does not imply a complete coverage of the space.

5. As stated earlier, we study abstract tasks that are inspired by the computations involved in creative tasks. Our study is not intended to capture the subjective aspects like interestingness (Schmidhuber, 2009) and other social, cultural and personal values integral to many creative tasks.

## B. Transformer Training Objectives

Let LMθ be our language model, parameterized by θ, for which LMθ(ˆsi = si; s<i) is the probability it assigns to the ith output sˆi being si, given as input a sequence s<i. Let (p, r) be a prefix-response pair. In standard next-token finetuning, we maximize the objective:

$${\mathcal{J}}_{\mathrm{next-token}}(\theta)=\mathbb{E}_{\mathcal{D}}\Big[\sum_{i=1}^{L_{\mathrm{resp}}}\log\mathrm{LM}_{\theta}\left({\hat{r}}_{i}=r_{i};\mathbf{p},\mathbf{r}_{<i}\right)\Big]$$

$$\left(2\right)$$
i(2)
In teacherless (multi-token) training (Monea et al., 2023; Bachmann & Nagarajan, 2024; Tschannen et al., 2023), we make use of an uninformative input string $ that simply corresponds to a series of dummy tokens $.

$${\mathcal{J}}_{\mathrm{multi-token}}(\theta)=\mathbb{E}_{\mathcal{D}}\Big[\sum_{i=1}^{L_{\mathrm{resp}}}\log\mathrm{LM}_{\theta}\left({\hat{r}}_{i}=r_{i};\mathbf{p},\mathbf{\hat{\Phi}}_{<i}\right)\Big]$$
$$(3)$$

i(3)

## C. Further Discussion C.1. Intuition For Seed-Conditioning

Seed-conditioning produces non-trivial algorithmic creativity even with greedy decoding. Perhaps, one could view the seed prefixes as a simpler alternative to varying the wordings of a prompt (Li et al., 2023; Lau et al., 2024; Naik et al., 2024; Li et al., 2022) or tuning a soft-prompt (Wang et al., 2024c), both of which are known to induce diversity. However, it is unexpected that the seeds provide any meaningful randomness at all, given they are arbitrarily picked with no semantic relationship to the training point. We do not understand this presently. Another pertinent question is to why seed-conditioning works better than temperature sampling in the settings it does. To this, we offer three potential explanations. The simplest explanation is that in the initial version of our paper, top-K sampling was enabled by default with K = 50 for the GPT-2 (86M) settings. This would have restricted the degree of freedom in the support induced by the output-layer randomness; thus, providing different seeds can help vary this support and produce more varying samples. However, even upon removing top-K sampling, we see gains in our exploratory creativity tasks for GPT-2 (86M) (after all their vocabulary is smaller than K = 50), and in all our Gemma v1 (2B) results. This is unexplained.

Thus, we speculatively put forth two more arguments. First, is a representational one. Fixing a random seed upfront may help the model flesh out (i.e., compute the tokens of) one thought per sample, as against maintaining a running set of multiple thoughts and computing a distribution over all their tokens at each step. A similar point is made in a concurrent position paper (Jahrens & Martinetz, 2025). The second argument is specific to next-token prediction on open-ended planning tasks: fixing a random seed upfront may help the model co-ordinate multiple interlocking random decisions in advance rather than deciding them on the fly. Finally, there are also optimization aspects of how seed-conditioning works that we do not understand (see §C.3). Regardless, it remains to be seen whether seed-conditioning is useful in tasks beyond the minimal ones we design.

## C.2. Effects Of Reasoning-Enhancing Methods.

Our argument is limited to learning open-ended tasks in a supervised manner which directly corresponds to how we perform pretraining or apply models for protein or drug discovery. While we do not comment on how well other powerful approaches like RL (DeepSeek-AI, 2025), chain-of-thought (CoT; Wei et al., 2022), and scaling test-time compute (OpenAI,
2024) would address the limitations we bring up, we make a few remarks. First, in cases where post-training only elicits pre-existing skills in the base model (Muennighoff et al., 2025), our limitations and suggestions about a next-token-trained base model remain relevant. Next, regardless of the benefits one may get from pre-training, our arguments imply that pre-training squanders its rich supervision, when it could be vastly compute- and data-efficient. Finally, there is a profound question as to whether exploration through sparse rewards can learn the creative process during training-time or whether articulation in the token space (with a thinking model) can efficiently maximize diversity during test-time (which may require brute-force enumeration of all candidates).

## C.3. Style Of Noise-Injection

Our technique of injecting noise into the model is somewhat different from how noise is introduced in traditional VAEs (Kingma & Welling, 2014) or GANs (Goodfellow et al., 2020), and this difference is worth noticing. In traditional approaches, although the model learns a noise-output mapping, this mapping is enforced only at a distribution level i.e., the distribution of noise vectors must map to a distribution of real vectors. However, in our approach we arbitrarily enforce what noise vector goes to what real datapoint, at a pointwise level. This raises the open questions of why seed-conditioning works in the first place - surprisingly, without breaking optimization or generalization - and whether there is a way to enforce it at distribution-level, and whether that can provide even greater improvements.

## C.4. In-Weights Vs In-Context Graphs For Combinational Creativity

Combinational creativity requires searching through known entities. In abstracting this, there is an interesting choice to be made as to whether the relevant search space is retrieved and spelled out in-context or whether it remains in-weights (like in Sibling Discovery and Triangle Discovery). We argue that the in-context version does not capture the creative skills required in many real-world tasks. For instance, discovering a fresh and surprising analogy necessitates noticing similarities from sufficiently distinct parts of one's vast, rich space of memory. Thus, the core challenge here lies in retrieving from the entirety of one's memory. If one were to faithfully simulate this an in-context version of this in a model, one would have to provide the entirety of the model's pretraining data in context.

## C.5. Examples Of Triangle Discovery

Although we presented this task as a more complex, higher-order counterpart to Sibling Discovery, we retrospectively identify some real-world examples that resemble the higher-order search skill involved in this task.

1. **Discovering contradictions:** Consider identifying non-trivial contradictions within (a large body) of knowledge (like a legal system, or a proof based on many lemmas, or the literature spanning many papers in a certain field). This may require identifying two or more facts that together result in an implication that contradicts another fact.

2. **Discovering feedback loops:** Fields like biology, ecology, climate science and economics may involve discovering non-trivial feedback loops. Unlike feedback loops where two events encourage each other, a non-trivial loop would be one where an Event A encourages Event B, that in turn encourages Event C that in turn encourages Event A.

3. **Antanaclasis:** An antanaclasis involves using a word in two different senses in a sentence, while still ensuring that each sense has a coherent relationship with the rest of the sentence. Consider Benjamin Franklin's quote, Your argument is sound, nothing but sound. Here, the two senses are sound1 as in "logically correct", sound2 as in "noise". This sentence encodes an pairwise relationship between three entities {argument, sound1,sound2}
individually. While the last two entities (the two senses) themselves must be related to each other (through the common word, sound), for a coherent sentence, both senses must also be appropriate descriptors for the first entity, argument. Thus, constructing this sentence requires searching through one's vocabulary to discover three words that satisfy these three relationships simultaneously.

4. **Word games:** Some word games require identifying a set of words that simultaneously have pairwise relationships with each other.

(a) For example, standard crosswords would require identifying sets of 4 or more words that have various simultaneous pairwise intersections in the letters used.

(b) Devising "& Lit." clues in cryptic crosswords are an altogether different, yet compelling example that require discovering a satisfying triangular relationship. Consider the clue "Some assassin in Japan" whose answer is Ninja. Here the phrase Some assassin in Japan participates in two senses. First, is the direct semantic sense as a definition of what a Ninja is. But there is a second, indirect