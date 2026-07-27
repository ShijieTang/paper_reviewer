# Arc-Agi Without Pretraining

Anonymous Author(s)
Affiliation Address email

## Abstract 15 **1 Introduction**

1 Conventional wisdom in the age of LLMs dictates that solving IQ-test-like puz2 zles from the ARC-AGI-1 benchmark requires capabilities derived from massive 3 pretraining. To counter this, we introduce *CompressARC*, a model without any 4 pretraining that solves 20% of evaluation puzzles by minimizing the description 5 length (MDL) of the target puzzle purely during inference time. The MDL endows 6 CompressARC with extreme generalization abilities typically unheard of in deep 7 learning. To our knowledge, CompressARC is the only deep learning method for 8 ARC-AGI where training happens only on a fraction of one sample: the target 9 inference puzzle itself, with the final solution information removed. Moreover, 10 CompressARC does not train on the pre-provided ARC-AGI "training set". Under 11 these extremely data-limited conditions, we do not ordinarily expect any puzzles to 12 be solvable at all. Yet CompressARC still solves a diverse distribution of creative 13 ARC-AGI puzzles, suggesting MDL to be an alternative, highly feasible way to 14 produce intelligence, besides conventional massive pretraining. 16 The ARC-AGI benchmark poses a uniquely challenging problem: to construct a system capable 17 of solving novel, abstract reasoning puzzles using only a handful of examples. [1] These puzzles 18 are intentionally designed to measure generalization, creativity, and pattern recognition, and have 19 historically resisted solutions by even the most powerful pretrained large language models (LLMs). 20 The most successful attempts have leaned heavily on massive datasets, fine-tuning, or test-time 21 augmentation. [2, 3, 4] However, one possible approach towards artificial general intelligence 22 (AGI) has remained surprisingly underexplored in practice: the principle of minimum description 23 length (MDL). [5] Closely related to Kolmogorov complexity [6], MDL frames intelligence as the 24 ability to compress information efficiently into a minimally sized program, that correctly outputs 25 the original information when run. Despite its elegant theoretical connection to generalization and 26 prediction, MDL has rarely been successfully implemented in deep learning as an alternative source 27 of intelligence to pretrained LLMs. In this work, we directly investigate the power of compression by 28 introducing *CompressARC*, a deep learning method that minimizes description length at inference 29 time: it has no prior training at all—and yet it still achieves modest performance on ARC-AGI. 30 CompressARC tries to harness MDL by using deep learning, a combination of techniques plagued 31 with incompatabilities and roadblocks. The main difficulty in using deep learning to minimize the 32 description length is that the description is a discrete program, and cannot be differentiated. Moreover, 33 the size of the program varies as we optimize over the program's code, running counter to gradient 34 descent's requirement of a fixed number of training parameters. Together, these two difficulties 35 make it nearly inconceivable to use gradient descent for searching the description space. As a result, 36 past MDL-based attempts to solve ARC-AGI have focused on search in (at least partially) discrete 37 program spaces. [7] The powerful expressive capacity of deep neural networks, requiring gradient 38 descent to achieve, has not yet been fully combined with the strong generalization abilities promised 39 by the MDL principle. These strengths are exactly what CompressARC has managed to conjoin. 40 The innovation that underlies CompressARC is a procedure for compiling the continuous information 41 stored in a tensor into a discrete code. This procedure is special in that we can track the expected 42 resulting code length from the perspective of the original continuous space, without ever having to 43 perform the compilation, all in a differentiable fashion. This affords us the ability to include neural 44 networks as part of the description, along with tensors representing their weights and inputs. The 45 entire problem of minimizing the discrete description length is then offloaded as a deep learning 46 task: the final procedure drawn in Figure 1. If we respect the restrictions imposed by the conversion 47 of MDL into a deep learning problem, then we may enjoy MDL's strong generalization abilities as 48 benefit: 49 - **No training time:** Since MDL requires us to start by having the target puzzle in hand, Compres50 sARC starts by skipping training time, to go to inference time immediately to first obtain the target 51 puzzle.

52 - **Inference time learning:** At this point, MDL dictates we minimize the description length, so 53 CompressARC must run gradient descent using the target puzzle during inference time, to produce 54 the solution.

55 - **Relaxed data requirement:** Since we expect to enjoy such strong generalization abilities endowed 56 by MDL, we don't bother loading any other puzzles into memory. The target puzzle just by itself is 57 already plenty of data. 58 Of course, this means CompressARC skips pretraining and leaves any training set puzzles unused. 59 Even so, the extreme generalization of MDL allows CompressARC to solve 20% of evaluation 60 puzzles and 34.75% of training puzzles, where we would ordinarily expect 0% from any traditional 61 deep learning method under these conditions. 62 The remaining sections describe the ARC-AGI benchmark (Section 2), how CompressARC works 63 (Section 3), CompressARC's architecture (Section 4), CompressARC's performance on ARC-AGI 64 (Section 5), our interpretation of CompressARC's solution to an example puzzle (Section 6), and our 65 conclusions (Section 7).

## 66 **2 Background: The Arc-Agi Benchmark**

67 ARC-AGI-1 is an artificial intelligence benchmark designed to test a system's ability to acquire new 68 skills from minimal examples. Each puzzle in the benchmark consists of a different hidden rule, 69 which the system must apply to an input colored grid to produce a ground truth target colored grid.

70 Several input-output grid pairs are given as examples to help the system to infer the hidden rule 71 in a puzzle. The system is allowed **two attempts** to guess the output grid correctly, i.e., getting 72 every single pixel color correct. The ARC Prize Foundation has launched competitions for machine 73 solutions to ARC-AGI-1, with upwards of **$1,000,000** in prizes. [2, 8] 74 There are 400 training puzzles are easier than the 400 evaluation puzzles, and are meant to help your 75 system learn the ideas of objectness, goal-directedness, numbers & counting, and basic geometry &
76 topology. **These training puzzles play no role in the operation of CompressARC, and we only** 77 **used them to inform our decisions of how to build CompressARC's architecture.** 78 The puzzles are designed so that **humans can reasonably find the answer, but machines should** 79 **have more difficulty**. The average human can solve 76.2% of the training set, and a human expert can 80 solve 98.5%. [9] Current methods for solving ARC-AGI focus primarily on tokenizing the puzzles 81 and arranging them in a sequence to prompt an LLM for a solution, or code that computes a solution. 82 [3] Top methods typically fine-tune on augmented training puzzles and larger alternative synthetic 83 puzzle datasets [10] and test-time training [4, 11]. Reasoning models have managed to get up to 84 87.5% on the semi-private evaluation set, albeit with astronomical amounts of compute. [12] 85 Please refer to Appendix K for more details about the ARC-AGI benchmark. An extended survey of 86 other related work is also included in Appendix H. 87 As of March 2025, the ARC Prize foundation has launched a new dataset and competition, ARC- 88 AGI-2, which is extremely similar in format to ARC-AGI-1. Since the research in this paper predates 89 the launch, this paper focuses solely on ARC-AGI-1, which in this paper we generally refer to as 90 ARC-AGI.

## 91 **3 Method**

92 We propose that MDL can serve as an effective framework for solving ARC-AGI puzzles. In MDL, a 93 more efficient (i.e., lower-bit) compression of a puzzle correlates with a more accurate solution. To 94 solve ARC-AGI puzzles, we design a system that transforms an incomplete puzzle into a completed 95 one—filling in the answers—by finding a compact representation (i.e., short program,) that when 96 run, reproduces the puzzle with any solution. The challenge is to algorithmically obtain this compact 97 program representation, given the puzzle.

98 Our key innovation is to notice that we can compile a sampling procedure from any continuous 99 random process into a short program, whose program length is very close to the KL divergence of this 100 process relative to some fixed reference process. This particular kind of compilation is made possible 101 by Relative Entropy Coding (REC) [13]. This fact means we can include randomized tensors in a 102 description, and count up their total description lengths as KL divergences which mirror the program 103 length of the compiled sampling procedures. We can even train the tensors with gradient descent to 104 minimize their description lengths as measured by KL terms. Gradient descent then can serve as a 105 description length minimizer in a space of deep learning based programs. Finally, as long as we know 106 that the description length is being minimized and we are able to extract the solution guess, there is 107 no actual need to run REC or compile any sampling procedures in practice. 108 In standard machine learning lingo, the operations CompressARC actually needs to perform are: 109 (with some simplifications, also see Figure 1) 110 1. We start at inference time, and we are given an ARC-AGI puzzle to solve. (e.g., puzzle in the 111 diagram below.)
112 2. We construct a neural network f (see Appendix C) designed for the puzzle's specifics (e.g., number 113 of examples, observed colors). The network takes random normal input z ∼ N(µ, Σ), and outputs 114 per-pixel color logit predictions across all the grids, including an answer grid (3 input-output 115 examples, for a total of 6 grids). Importantly, fθ is equivariant to common augmentations—such 116 as reordering input-output pairs (including the answer's pair), color permutations, and spatial 117 rotations/reflections. 118 3. We initialize the network weights θ and set the parameters µ and Σ for the z distribution.

119 4. We jointly optimize θ, µ, Σ to minimize the sum of cross-entropies over the known grids (5 of 120 them,) ignoring the answer grid. A KL divergence penalty keeps N(µ, Σ) close to N(0, 1), as in 121 a VAE. 122 5. Since the generated answer grid is stochastic due to the randomness in z, we save the answer grids 123 throughout training and choose the most frequently occuring one as our final prediction. 124 The short program that we would compile the weight θ and input z distributions into, in trying to 125 minimize the program code length, looks like the following: 126 z = sample_normal(N(0,I), <seed_z>) 127 weights = <insert weights here> 128 puzzle_and_solution_logits = neural_net(z, weights) 129 puzzle_and_solution = sample_categorical(puzzle_and_solution_logits, <seed_error>) 135 Appendix A contains a more elaborate explanation of why we picked this particular program as our 136 candidate shortest program.

## 137 **4 Architecture**

138 We designed our own neural network architecture for decoding the latents z into ARC-AGI puzzles, 139 illustrated in Figure 2. The most important feature of our architecture is it's equivariances, which are 140 symmetry rules dictating that whenever the input z undergoes a transformation, the output ARC-AGI 141 puzzle must also transform the same way. Some example transformations include reordering of 142 input/output pairs, shuffling colors, flips, rotations, and reflections of grids. 143 The data format of z is what we call a "multitensor", which is a bucket of tensors that each may or may 144 not have certain dimensions such as example, color, height, width dimensions, which transformations 145 can be applied to. All the equivariances can be described in terms of how they change a multitensor. 146 More details on multitensors are in Appendix B
Figure 2: Overall structure of CompressARC's equivariant neural network. There were too many equivariances for us to consider at once, so we decided to make a **base architecture that's fully** symmetric, and break unwanted symmetries one by one by **adding asymmetric layers** to give it specific non-equivariant abilities (listed later in Appendix G).

147 The architecture is complicated and has many types of layers that we designed to have inductive 148 biases that are useful for solving the given training puzzles. The training puzzles play no role in our 149 work other than in this way and in our evaluations. The full architecture consists of the following 150 layers, which are each described in the Appendix:
130 where <seed_z> and <seed_error> are randomization seeds picked by REC to force z ∼ N(µ, Σ) 131 and correct final puzzle sampling, with the seeds being approximately KL(N(µ, Σ)||N(0, I)) and 132 CrossEntropyLoss(puzzle_and_solution_logits, true_puzzle, reduction='sum') 133 bits long, respectively. Our inference-time training setup and chosen loss function serves entirely to 134 shorten the seeds needed by this compiled program, in order to optimize it for Solomoff induction. 151 - Begin with parameters of the z distribution 152 - Decoding Layer, Appendix C.1 153 - Repeat 4 times: 161 - Normalization Layer, Appendix C.7 162 - Linear Heads, Appendix C.8

## 163 **5 Results**

164 CompressARC solves 20% of evaluation set puzzles and 34.75% of training set puzzles if given 2000 165 steps per puzzle, as shown in Tables 1 and 2, and Figure 3.

Figure 3: CompressARC's puzzle solve accuracy as a function of the number of steps of inference time learning it is given, for various numbers of allowed attempts (pass@n). The official benchmark is reported with 2 allowed attempts, which is why we report 20% on the evaluation set.

Table 1: CompressARC's puzzle solve accuracy on the training set as a function of the number of steps of inference time learning it is given, for various numbers of allowed attempts (pass@n). The official benchmark is reported with 2 allowed attempts, which is why we report 20% on the evaluation set. Timing is reported for an NVIDIA RTX 4070 GPU.

Training Iteration Time Pass@1 Pass@2 Pass@5 Pass@10 Pass@100 Pass@1000

100 6 h 1.00% 2.25% 3.50% 4.75% 6.75% 6.75%

200 13 h 11.50% 14.25% 16.50% 18.25% 23.25% 23.50%

300 19 h 18.50% 21.25% 23.50% 26.75% 31.50% 32.50% 400 26 h 21.00% 25.00% 28.75% 31.00% 36.00% 37.50% 500 32 h 23.00% 27.50% 31.50% 33.50% 39.25% 40.75% 750 49 h 28.00% 30.50% 34.00% 36.25% 42.75% 44.50%

1000 65 h 28.00% 31.75% 35.50% 37.75% 43.75% 46.50% 1250 81 h 29.00% 32.25% 37.00% 39.25% 45.50% 49.25% 1500 97 h 29.50% 33.00% 38.25% 40.75% 46.75% 51.75% 2000 130 h 30.25% 34.75% 38.25% 41.50% 48.50% 52.75%

154 - Multitensor Communication Layer (Upwards), Appendix C.2 155 - Softmax Layer, Appendix C.3 156 - Directional Cummax Layer, Appendix C.4 157 - Directional Shift Layer, Appendix C.4 158 - Directional Communication Layer, Appendix C.5 159 - Nonlinear Layer, Appendix C.6 160 - Multitensor Communication Layer (Downwards), Appendix C.2

Table 2: CompressARC's puzzle solve accuracy on the evaluation set, reported the same way as in Table 1.

Training Iteration Time Pass@1 Pass@2 Pass@5 Pass@10 Pass@100 Pass@1000

100 7 h 0.75% 1.25% 2.25% 2.50% 3.00% 3.00% 200 14 h 5.00% 6.00% 7.00% 7.75% 12.00% 12.25%

300 21 h 10.00% 10.75% 12.25% 13.25% 15.50% 16.25%

400 28 h 11.75% 13.75% 16.00% 17.00% 19.75% 20.00% 500 34 h 13.50% 15.00% 17.75% 19.25% 20.50% 21.50% 750 52 h 15.50% 17.75% 19.75% 21.50% 22.75% 25.50%

1000 69 h 16.75% 19.25% 21.75% 23.00% 26.00% 28.75% 1250 86 h 17.00% 20.75% 23.00% 24.50% 28.25% 30.75% 1500 103 h 18.25% 21.50% 24.25% 25.50% 29.50% 31.75% 2000 138 h 18.50% 20.00% 24.25% 26.00% 31.25% 33.75%

## 166 **5.1 What Puzzles Can And Can'T We Solve?**

167 **CompressARC tries to use its abilities to figure out as much as it can, until it gets bottlenecked** 168 **by one of it's inabilities.**
169 For example, puzzle 28e73c20 in the training set requires extension of a pattern from the edge towards 170 the middle, as shown in Figure 11a in the Appendix. Given the layers in it's network, CompressARC 171 is generally able to extend patterns for short ranges but not long ranges. So, it does the best that 172 it can, and correctly extends the pattern a short distance before guessing at what happens near the 173 center (Figure 11b, Appendix). Appendix G includes a list of which abilities we have empirically 174 seen CompressARC able to and not able to perform.

## 175 **6 Case Study: Color The Boxes**

176 In this puzzle (Puzzle 272f95fa, Figure 4), you must color sections depending on which side of the 177 grid the section is on. We call this puzzle "Color the Boxes".

## 191 **6.1 Solution Analysis**

192 So how does CompressARC learn to solve Color the Boxes? We can look at the representations 193 stored in z to find out.

194 Since z is a multitensor, each of the tensors it contains produces an additive contribution to the total 195 KL for z. By looking at the per-tensor contributions (see Figure 5b), we can determine which tensors 196 in z code for information that is used to represent the puzzle. 197 All the tensors fall to zero information content during training, except for four tensors. In some 198 replications of this experiment, we saw one of these four necessary tensors fall to zero information 199 content, and CompressARC typically does not recover the correct answer after that. Here we are 200 showing a lucky run where the [color, direction, channel] tensor almost falls but gets picked up 200 201 steps in, which is right around when the samples from the model begin to show the correct colors in 202 the correct boxes. 203 We can look at the average output of the decoding layer (explained in Appendix C.1) corresponding 204 to individual tensors of z, to see what information is stored there (see Figure 6). Each tensor contains 205 a vector of dimension n_channels for various indices of the tensor. Taking the PCA of these vectors 206 reveals some number of activated components, telling us how many pieces of information are coded 207 by the tensor. 178 **Human Solution:** We first realize that the input is divided into boxes, and the boxes are still there in 179 the output, but now they're colored. We then try to figure out which colors go in which boxes. First, 180 we notice that the corners are always black. Then, we notice that the middle is always magenta. And 181 after that, we notice that the color of the side boxes depends on which direction they are in: red for 182 up, blue for down, green for right, and yellow for left. At this point, we copy the input over to the 183 answer grid, then we color the middle box magenta, and then color the rest of the boxes according to 184 their direction. 185 **CompressARC Solution:** Table 3 shows CompressARC's learning behavior over time. After 186 CompressARC is done learning, we can deconstruct it's learned z distribution to find that it codes for 187 a color-direction correspondence table and row/column divider positions (Figure 6). 188 During training, the reconstruction error fell extremely quickly. It remained low on average, but 189 would spike up every once in a while, causing the KL from z to bump upwards at these moments, as 190 shown in Figure 5a.

| Table 3: CompressARC learning the solution for Color the Boxes, over time.   |                            |                        |
|------------------------------------------------------------------------------|----------------------------|------------------------|
| Learning steps                                                               | What is CompressARC doing? | Sampled solution guess |
| CompressARC's network outputs an answer grid (sample) with light blue rows/columns wherever the input has the same. It has noticed that all the other input-output pairs in the puzzle exhibit this correspondence. It doesn't know how the other output pixels are assigned colors; an exponential moving average of the network output (sample average) shows the network assigning mostly the same average color to non-light-blue pixels.                                                                              |                            |                        |
| 50                                                                           | The network outputs a grid where nearby pixels have similar colors. It has likely noticed that this is common among all the outputs, and is guessing that it applies to the answer too.                            |                        |
| 150                                                                          | The network output now shows larger blobs of colors that are cut off by the light blue borders. It has noticed the common usage of borders to demarcate blobs of colors in other outputs, and applies the same idea here. It has also noticed black corner blobs in other given outputs, which the network imitates.                            |                        |
| 200                                                                          | The network output now shows the correct colors assigned to boxes of the correct direction from the center. It has realized that a single color-to-direction mapping is used to pick the blob colors in the other given outputs, so it imitates this mapping. It is still not the best at coloring within the lines, and it's also confused about the center blob, probably because the middle does not correspond to a direction. Nevertheless, the average network output does show a tinge of the correct magenta color in the middle, meaning the network is catching on.                            |                        |
| 350                                                                          | The network is as refined as it will ever be. Sometimes it will still make a mistake in the sample it outputs, but this uncommon and filtered out.                            |                        |
| 1500                                                                         |                            |                        |

## 208 **7 Discussion**

209 The prevailing reliance of modern deep learning on high-quality data has put the field in a chokehold 210 when applied to problems requiring intelligent behavior that have less data available. This is espe211 cially true for the data-limited ARC-AGI benchmark, where LLMs trained on specially augmented, 212 extended, and curated datasets dominate. In the midst of this circumstance, we built CompressARC,
213 which not only uses no training data at all, but forgoes the entire process of pretraining altogether.

214 One should intuitively expect this to fail and solve no puzzles at all, but by applying MDL to the target 215 puzzle during inference time, CompressARC solves a surprisingly large portion of ARC-AGI-1. 216 CompressARC's theoretical underpinnings come from minimizing the description length of the target 217 puzzle. While other MDL search strategies have been scarce due to the intractablly large search 218 space of possible programs, CompressARC explores a simplified, neural network-based search space 219 through gradient descent. Though CompressARC's architecture is heavily engineered, it's incredible 220 ability to generalize from as low as two demonstration input/output pairs puts it in an entirely new 221 regime of generalization for ARC-AGI. 222 We challenge the assumption that intelligence must arise from massive pretraining and data, showing 223 instead that clever use of MDL and compression principles can lead to surprising capabilities. We 224 use CompressARC a proof of concept to demonstrate that modern deep learning frameworks can be 225 melded with MDL to create a possible alternative, complimentary route to AGI.

## 226 **References**

227 [1] François Chollet. On the measure of intelligence, 2019. 228 [2] Francois Chollet, Mike Knoop, Gregory Kamradt, and Bryan Landers. Arc prize 2024: Technical 229 report, 2025.

230 [3] Ryan Greenblatt. Getting 50% (sota) on arc-agi with gpt-4o. https://redwoodresearch.

231 substack.com/p/getting-50-sota-on-arc-agi-with-gpt, 2024. Accessed: 2025-05232 12. 233 [4] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, and Moritz Hardt. Test-time 234 training with self-supervision for generalization under distribution shifts, 2020. 235 [5] J. Rissanen. Modeling by shortest data description. *Automatica*, 14(5):465–471, 1978. 236 [6] A.N. Kolmogorov. On tables of random numbers. *Theoretical Computer Science*, 207(2):387– 237 395, 1998.

238 [7] Sébastien Ferré. Madil: An mdl-based framework for efficient program synthesis in the arc 239 benchmark, 2025. 240 [8] Mike Knoop. ARC Prize 2024 Winners & Technical Report Published - arcprize.org. https:
241 //arcprize.org/blog/arc-prize-2024-winners-technical-report, 2024. [Ac-242 cessed 12-05-2025].

243 [9] Solim LeGris, Wai Keen Vong, Brenden M. Lake, and Todd M. Gureckis. H-arc: A robust 244 estimate of human performance on the abstraction and reasoning corpus benchmark, 2024.

245 [10] Wen-Ding Li, Keya Hu, Carter Larsen, Yuqing Wu, Simon Alford, Caleb Woo, Spencer M.

246 Dunn, Hao Tang, Michelangelo Naim, Dat Nguyen, Wei-Long Zheng, Zenna Tavares, Yewen 247 Pu, and Kevin Ellis. Combining induction and transduction for abstract reasoning, 2024. 248 [11] Guillermo Barbadillo. Solution summary for arc24. https://ironbar.github.io/arc24/ 249 05_Solution_Summary/, 2024. Accessed: 2025-05-12. 250 [12] François Chollet. Openai o3 breakthrough high score on arc-agi-pub. https://arcprize. 251 org/blog/oai-o3-pub-breakthrough, 2024. Accessed: 2025-05-12. 252 [13] Gergely Flamich, Marton Havasi, and José Miguel Hernández-Lobato. Compressing images by 253 encoding their latent representations with relative entropy coding, 2021. 254 [14] Thomas M. Cover and Joy A. Thomas. *Elements of Information Theory*. John Wiley & Sons, 255 Inc., 2006. 256 [15] Peter D. Grunwald and Paul M. B. Vitanyi. Algorithmic information theory, 2008. 257 [16] James Irvine and David Harle. *Data Communications and Networks: An Engineering Approach*. 258 Wiley, New York, 1 edition, 2001. Hardcover. 259 [17] C. E. Shannon. A mathematical theory of communication. *The Bell System Technical Journal*, 260 27(3):379–423, 1948. 263 [19] Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2022.

264 [20] Marcus Hutter. Hutter prize for lossless compression of human knowledge. https://prize.

265 hutter1.net/, 2006. Accessed: 2025-05-12. 266 [21] R.J. Solomonoff. A formal theory of inductive inference. part i. *Information and Control*, 267 7(1):1–22, 1964. 268 [22] Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines, 2014.

261 [18] G. G. Langdon. An introduction to arithmetic coding. *IBM Journal of Research and Develop-*262 *ment*, 28(2):135–149, 1984.

269 [23] Casper Kaae Sønderby, Tapani Raiko, Lars Maaløe, Søren Kaae Sønderby, and Ole Winther. 270 Ladder variational autoencoders, 2016. 271 [24] C. Shannon. The zero error capacity of a noisy channel. *IRE Transactions on Information* 272 *Theory*, 2(3):8–19, 1956. 273 [25] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, 274 Shakir Mohamed, and Alexander Lerchner. beta-VAE: Learning basic visual concepts with a 275 constrained variational framework. In *International Conference on Learning Representations*, 276 2017. 277 [26] Arash Vahdat and Jan Kautz. Nvae: A deep hierarchical variational autoencoder, 2021.

278 [27] Michael Hodel. Domain specific language for the abstraction and reasoning corpus. https: 279 //github.com/michaelhodel/arc-dsl/blob/main/arc_dsl_writeup.pdf, 2024. Ac280 cessed: 2025-05-12. 281 [28] Victor Vikram Odouard. Arc-solution_documentation. https://github.com/ 282 victorvikram/ARC-icecuber/blob/master/ARC-solution_documentation.pdf, 283 2024. Accessed: 2025-05-12. 284 [29] Clément Bonnet and Matthew V Macfarlane. Searching latent program spaces, 2024. 285 [30] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, 286 Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2023.

287 [31] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image 288 recognition, 2015. 289 [32] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai 290 Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer 291 architecture, 2020. 292 [33] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus), 2023. 293 [34] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, 294 and Alexander Smola. Deep sets, 2018. 295 [35] Taco S. Cohen and Max Welling. Group equivariant convolutional networks, 2016.

296 [36] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, 297 Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. 298 [37] Vinod Kumar Chauhan, Jiandong Zhou, Ping Lu, Soheila Molaei, and David A. Clifton. A brief 299 review of hypernetworks in deep learning. *Artificial Intelligence Review*, 57(9), August 2024. 300 [38] Shiqing Fan, Liu Liying, and Ye Luo. An alternative practice of tropical convolution to 301 traditional convolutional neural networks, 2021. 302 [39] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation 303 learning, 2018.

## 304 **A Optimality Of Our Candidate Shortest Program**

305 It isn't obvious how we get from trying to minimize the description length to the method we ended 306 up using. The derivation of our algorithm takes us on a detour through information theory [14], 307 algorithmic information theory [15], and coding theory [16], with machine learning only making an 308 appearance near the end.

## 309 **A.1 A Primer On Lossless Information Compression**

310 In information theory, lossless information compression is about trying to represent some informa311 tion in as few bits as possible, while still being able to reconstruct that information from the bit 312 representation. [17] This type of problem is abstracted as follows: 313 - A source produces some symbol x from some process that generates symbols from a probability 314 distribution p(x). 315 - A compressor/encoder E must map the symbol x to a string of bits s. 316 - A decompressor/decoder D must exactly map s back to the original symbol x. 317 The goal in lossless information compression is to use p to construct functions (*E, D*) which are 318 bit-efficient, (i.e., that minimize the expected length of s,) without getting any symbols wrong. The optimal decompressor D∗
319 also plays a role in a program that is the shortest possible (up to additive 320 constants in program length) that computes x, in expectation over x drawn from p: 321 s = <string of bits> 322 x = D*(s) 323 This reduces MDL to the problem of lossless information compression. In our case, the symbol x is 324 the ARC-AGI dataset (many puzzle + answer pairs), and we may want to figure out what D* is using 325 knowledge of p, and what s is when given x. Except, we won't have the answers (only the puzzles)
326 in x, and we don't actually know p, since it's hard to model the intelligent process of puzzle ideation 327 in humans.

## 328 **A.2 One-Size-Fits-All Compression**

329 To build an efficient lossless compression scheme, you might think we need to know what p is, but 330 we argue that it doesn't really matter since we can make a one-size-fits-all compressor. It all hinges 331 on the following assumption: 332 **There exists some practically implementable, bit efficient compression system** (*E, D*) for ARC- 333 AGI datasets x **sampled from** p. 334 If this were false, our whole idea of solving ARC-AGI with compression will be doomed even if we 335 knew p anyways, so we might as well make this assumption.

Our one-size-fits-all compressor (E′, D′
336 ) is built without knowing p, and it is almost just as bit337 efficient as the original (E, D):
- E′
338 observes symbol x, picks a program f and input s to minimize len(f) + len(s) under the 339 constraint that running the program makes f(s) = x, and then sends the pair (*f, s*).

- D′
340 is just a program executor that executes f on s, correctly producing x.

It is possible to prove with algorithmic information theory that (E′, D′
341 ) achieves a bit efficiency at 342 most len(f) bits worse than the bit efficiency of (*E, D*), where f is the code for implementing D.

343 [15] But since compression is practically implementable, the code for D should be simple enough for 344 a human engineer to write, so len(f) must be short, meaning our one-size-fits-all compressor will be 345 close to the best possible bit efficiency.

Ironically, the only problem with using this to solve ARC-AGI is that implementing E′
346 is not practical, since E′
347 needs to minimize the length of a program-input pair (f, s) under partial fixed 348 output constraint f(s)puzzle = xpuzzle.

## 349 **A.3 Neural Networks To The Rescue**

350 To avoid searching through program space, we just pick a program f for a small sacrifice in bit 351 efficiency. We hope the diversity of program space can be delegated to diversity in input s space 352 instead. Specifically, we write a program f that runs the forward pass of a neural network, where 353 s = (*θ, z, ϵ*) are the weights, inputs, and corrections to the outputs of the neural network. Then, we 354 can use gradient descent to "search" over s.

This restricted compression scheme uses Relative Entropy Coding (REC) [13]
1 355 to encode noisy 356 weights θ and neural network inputs z into bits sθ and sz, and arithmetic coding [18] to encode 357 output error corrections ϵ into bits sϵ, to make a bit string s consisting of three blocks (sθ, sz, sϵ).

358 The compression scheme runs as follows:
359 - The decoder runs θ = REC-decode(sθ), z = REC-decode(sz), logits = Neural-Net(*θ, z*), and 360 x = Arithmetic-decode(sϵ, logits).

361 - The encoder trains θ and z to minimize the total code length E[len(s)]. sϵ is fixed by arithmetic 362 coding to guarantee correct decoding. To calculate the three components of the loss E[len(s)] in a 363 differentiable way, we refer to the properties of REC and arithmetic coding:
364 - It turns out that the ϵ code length E[len(sϵ)] is equal to the total crossentropy error on all the 365 given grids in the puzzle. 366 - REC requires us to fix some reference distribution qθ, and also add noise to θ, turning it into 367 a distribution pθ. Then, REC allows you to store noisy θ using a code length of E[len(sθ)] =
KL(pθ||qθ) = Eθ∼pθ 368 [log(pθ(θ)/qθ(θ))] bits. We will choose to fix qθ = N(0*, I/*2λ) for large λ, such that the loss component E[len(sθ)] ≈ λ|θ| 369 2 + const is equivalent to regularizing the 370 decoder. 371 - We must also do for z what we do for θ, since it's also represented using REC. We will 372 choose to fix qz = N(0, I), so the code length of z is E[len(sz)] = KL(pz||qz) =
Ez∼pz 373 [log(pz(z)/qz(z))].

374 We can compute gradients of these code lengths via the reparameterization trick. [19]
375 At this point, we observe that the total code length for s that we described is actually the VAE loss with decoder regularization (= KL for z + reconstruction error + regularization).2 376 Likewise, if we 377 port the rest of what we described above (plus modifications regarding equivariances and inter-puzzle 378 independence, and ignoring regularization) into typical machine learning lingo, we get the previous 379 description of CompressARC from Section 3.

## 380 **B Multitensors**

381 The actual data (z, hidden activations, and puzzles) passing through our layers comes in a format that 382 we call a "**multitensor**", which is just a bucket of tensors of various shapes, as shown in Figure 7. 383 All the equivariances we use can be described in terms of how they change a multitensor.

384 Most common classes of machine learning architectures operate on a single type of tensor with 385 constant rank. LLMs operate on rank-3 tensors of shape [n_batch, n_tokens, n_channels],
386 and Convolutional Neural Networks (CNNs) operate on rank-4 tensors of shape 1A lot of caveats/issues are introduced by using REC. The code length when using REC only behaves in some limits and expectations, there may be a small added constant to the code length, the decoding may be approximate, etc. We're not up to date with the current literature, and we're ignoring all the sticky problems that may arise and presuming that they are all solved. We will never end up running Relative Entropy Coding anyways, so it doesn't matter that it takes runtime exponential in the code length. We only need to make use of the the fact that such algorithms exist, not that they run fast, nor that we can implement them, in order to derive our method.

2We penalize the reconstruction error by 10x the KL for z, in the total KL loss. This isn't detrimental to the measurement of the total KL because the KL term for z can absorb all of the coded information from the reconstruction term, which can then go to zero. Since the term for z is not penalized by any extra factor, the total KL we end up with is then unaffected. We believe this empirically helps because the Gaussians we use for z are not as efficient for storing bits that can be recovered, as the categorical distributions that define the log likelihood in the reconstruction error. Forcing all the coded bits into one storage mode removes pathologies introduced by multiple storage modes.

387 [n_batch, n_channels, height, width]. Our multitensors are a set of varying-rank ten388 sors of unique type, whose dimensions are a subset of a rank-6 tensor of shape 389 [n_examples, n_colors, n_directions, height, width, n_channels], as illustrated in Figure 7. We 390 always keep the channel dimension, so there are at most 32 tensors in each multitensor. We also 391 maintain several rules (see Appendix D.1) that determine whether a tensor shape is "legal" or not, 392 which reduces the number of tensors in a multitensor to 18.

| Dimension   | Description                                                                                                                                                                            |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Example     | Number of examples in the ARC-AGI puzzle, including the one with held-out answer                                                                                                       |
| Color       | Number of unique colors in the ARC-AGI puzzle, not including black, see Appendix E.2                                                                                                                                                                                        |
| Direction   | 8                                                                                                                                                                                      |
| Height      | Determined when preprocessing the puzzle, see Appendix E.1                                                                                                                             |
| Width       | Determined when preprocessing the puzzle, see Appendix E.1                                                                                                                             |
| Channel     | In the residual connections, the size is 8 if the direction dimension is included, else 16. Within layers it is layer-dependent. Table 4: Size conventions for multitensor dimensions. |

393 To give an idea of how a multitensor stores data, an ARC-AGI puzzle can be represented by 394 using the [example, color, height, width, channel] tensor, by using the channel dimension to select 395 either the input or output grid, and the height/width dimensions for pixel location, a one hot vector 396 in the color dimension, specifying what color that pixel is. The [example, height, channel] and 397 [example, width, channel] tensors can similarly be used to store masks representing grid shapes for 398 every example for every input/output grid. All those tensors are included in a single multitensor that 399 is computed by the network just before the final linear head (described in Appendix C.8). 400 When we apply an operation on a multitensor, we by default assume that all non-channel dimensions 401 are treated identically as batch dimensions by default. The operation is copied across the indices 402 of dimensions unless specified. This ensures that we keep all our symmetries intact until we use a 403 specific layer meant to break a specific symmetry.

404 A final note on the channel dimension: usually when talking about a tensor's shape, we will not even 405 mention the channel dimension as it is included by default.

## 406 **C Layers In The Architecture** 407 **C.1 Decoding Layer**

- A learned target multiscalar, called the "target capacity".3 415 The decoding layer will output z whose information content per tensor is close to the target capacity,4 416
- learned per-element means for z, 5 417 418 - learned per-element capacity adjustments for z.

## 430 **C.2 Multitensor Communication Layer**

431 This layer allows different tensors in a multitensor to interact with each other.

## 439 **C.3 Softmax Layer**

432 First, the input from the residual stream passes through per-tensor projections to a fixed size (8 for 433 downwards communication and 16 for upwards communication). Then a message is sent to every 434 other tensor that has at least the same dimensions for upwards communication, or at most the same 435 dimensions for downwards communication. This message is created by either taking means along 436 dimensions to remove them, or unsqueezing+broadcasting dimensions to add them, as in Figure 8.

437 All the messages received by every tensor are summed together and normalization is applied. This 438 result gets up-projected back and then added to the residual stream. 408 This layer's job is to sample a multitensor z and bound its information content, before it is passed 409 to the next layer. This layer and outputs the KL divergence between the learned z distribution and 410 N(0, I). Penalizing the KL prevents CompressARC from learning a distribution for z that memorizes 411 the ARC-AGI puzzle in an uncompressed fashion, and forces CompressARC to represent the puzzle 412 more succinctly. Specifically, it forces the network to spend more bits on the KL whenever it uses z 413 to break a symmetry, and the larger the symmetry group broken, the more bits it spends. 414 This layer takes as input:
We begin by normalizing the learned per-element means for z.

6 419 Then, we figure out how much 420 Gaussian noise we must add into every tensor to make the AWGN channel capacity [17] equal to the 421 target capacity for every tensor (including per-element capacity adjustments). We apply the noise to sample z, keeping unit variance of z by rescaling.7 422 423 We compute the information content of z as the KL divergence between the distribution of this sample 424 and N(0, 1).

Finally, we postprocess the noisy z by scaling it by the sigmoid of the signal-to-noise ratio.8 425 This 426 ensures that z is kept as-is when its variance consists mostly of useful information and it is nearly 427 zero when its variance consists mostly of noise. All this is done 4 times to make a channel dimension 428 of 4. Then we apply a projection (with different weights per tensor in the multitensor, i.e., per-tensor 429 projections) mapping the channel dimension up to the dimension of the residual stream.

443 over these subsets of dimensions, and concatenates all the softmaxxed results together in the channel 444 dimension. The output dimension varies across different tensors in the multitensor, depending on 445 their tensor rank. A pre-norm is applied, and per-tensor projections map to and from the residual 446 stream. The layer has input channel dimension of 2.

## 447 **C.4 Directional Cummax/Shift Layer**

456 The directional cummax layer takes the eight indices of the direction dimension, treats each slice as 457 corresponding to one direction (4 cardinal, 4 diagonal), performs a cumulative max in the respective 458 direction for each slice, does it in the opposite direction for half the channels, and stacks the slices 459 back together in the direction dimension. An illustration is in Figure 9. The slices are rescaled to 460 have min −1 and max 1 before applying the cumulative max. 461 The directional shift layer does the same thing, but for shifting the grid by one pixel instead of 462 applying the cumulative max, and without the rescaling. 463 Some details: 464 - Per-tensor projections map to and from the residual stream, with pre-norm. 465 - Input channel dimension is 4. 466 - These layers are only applied to the [example, color, direction, height, width, channel] and 467 [example, direction, height, width, channel] tensors in the input multitensor.

## 468 **C.5 Directional Communication Layer**

469 By default, the network is equivariant to permutations of the eight directions, but we only want 470 symmetry up to rotations and flips. So, this layer provides a way to send information between two 471 slices in the direction dimension, depending on the angular difference in the two directions. This 472 layer defines a separate linear map to be used for each of the 64 possible combinations of angles, 473 but the weights of the linear maps are minimally tied such that the directional communication layer 448 The directional cummax and shift layers allow the network to perform the non-equivariant cummax 449 and shift operations in an equivariant way, namely by applying the operations once per direction, and 450 only letting the output be influenced by the results once the directions are aggregated back together 451 (by the multitensor communication layer). These layers are the sole reason we included the direction 452 dimension when defining a multitensor: to store the results of directional layers and operate on 453 each individually. Of course, this means when we apply a spatial equivariance transformation, we 454 must also permute the indices of the direction dimension accordingly, which can get complicated 455 sometimes.

474 is equivariant to reflections and rotations. This gets complicated really fast, since the direction 475 dimension's indices also permute when equivariance transformations are applied. Every direction slice in a tensor accumulates it's 8 messages, and adds the results together.10 476 477 For this layer, there are per-tensor projections to and from the residual stream with pre-norm. The 478 input channel dimension is 2.

## 479 **C.6 Nonlinear Layer**

480 We use a SiLU nonlinearity with channel dimension 16, surrounded by per-tensor projections with 481 pre-norm.

## 482 **C.7 Normalization Layer**

483 We normalize all the tensors in the multitensor, using means and variances computed across all 484 dimensions except the channel dimension. Normalization as used within other layers also generally 485 operates this way.

## 486 **C.8 Linear Heads**

487 We must take the final multitensor, and convert it to the format of an ARC-AGI puzzle. More 488 specifically, we must convert the multitensor into a distribution over ARC-AGI puzzles, so that we 489 can compute the log-likelihood of the observed grids in the puzzle.

490 The colors of every pixel for every example for both input and output, have logits defined by the 491 [example, color, height, width, channel] tensor, with the channel dimension linearly mapped down to a size of 2, representing the input and output grids.11 492 The log-likelihood is given by the crossentropy, 493 with sum reduction across all the grids. 494 For grids of non-constant shape, the [example, height, channel] and [example, width, channel] tensors 495 are used to create distributions over possible contiguous rectangular slices of each grid of colors, 496 as shown in Figure 10. Again, the channel dimension is mapped down to a size of 2 for input and 497 output grids. For every grid, we have a vector of size [width] and a vector of size [height]. The log 498 likelihood of every slice of the vector is taken to be the sum of the values within the slice, minus 499 the values outside the slice. The log likelihoods for all the possible slices are then normalized to 500 have total probability one, and the colors for every slice are given by the color logits defined in the 501 previous paragraph. 502 With the puzzle distribution now defined, we can now evaluate the log-likelihood of the observed target puzzle, to use as the reconstruction error.12 503

## 504 **D Other Architectural Details** 505 **D.1 Rules For Legal Multitensors**

506 1. At least one non-example dimension must be included. Examples are not special for any reason 507 not having to do with colors, directions, rows, and columns. 508 2. If the width or height dimension is included, the example dimension should also be included. 509 Positions are intrinsic to grids, which are indexed by the example dimension. Without a grid it 510 doesn't make as much sense to talk about positions.

## 511 **D.2 Weight Tying For Reflection/Rotation Symmetry**

512 When applying a different linear layer to every tensor in a multitensor, we have a linear layer for 513 tensors having a width but not height dimension, and another linear layer for tensors having a height 514 but not width dimension. Whenever this is the case, we tie the weights together in order to preserve 515 the whole network's equivariance to diagonal reflections and 90 degree rotations, which swap the 516 width and height dimensions. 517 The softmax layer is not completely symmetrized because different indices of the output correspond 518 to different combinations of dimension to softmax over. Tying the weights properly would be a bit 519 complicated and time consuming for the performance improvement we expect, so we did not do this.

## 520 **D.3 Training**

521 We train for 2000 iterations using Adam, with learning rate 0.01, β1 of 0.5, and β2 of 0.9.

## 522 **E Preprocessing** 523 **E.1 Output Shape Determination**

524 The raw data consists of grids of various shapes, while the neural network operates on grids of 525 constant shape. Most of the preprocessing that we do is aimed towards this shape inconsistency 526 problem. 527 Before doing any training, we determine whether the given ARC-AGI puzzle follows three possible 528 shape consistency rules: 529 1. The outputs in a given ARC-AGI puzzle are always the same shape as corresponding inputs. 530 2. All the inputs in the given ARC-AGI puzzle are the same shape. 531 3. All the outputs in the given ARC-AGI puzzle are the same shape. 537 The largest width and height that is given or predicted, are used as the size of the multitensor's width 538 and height dimensions. 539 The predicted shapes are also used as masks when performing the multitensor communication, 540 directional communication and directional cummax/shift layers. We did not apply masks for the 541 other layers because of time constraints and because we do not believe it will provide for much of a performance improvement.13 542

## 543 **E.2 Number Of Colors**

544 We notice that in almost all ARC-AGI puzzles, colors that are not present in the puzzle are not present 545 in the true answers. Hence, any colors that do not appear in the puzzle are not given an index in the 546 color dimension of the multitensor. 547 In addition, black is treated as a special color that is never included in the multitensor, since it 548 normally represents the background in many puzzles. When performing color classification, a tensor 549 of zeros is appended to the color dimension after applying the linear head, to represent logits for the 550 black color.

## 551 **F Postprocessing**

558 1. Find the most commonly sampled answer.

552 Postprocessing primarily deals with denoising the answers sampled from the network. This is 553 complicated by the variable shape grids present in some puzzles. 554 Generally, when we sample answers from the network by taking the logits of the 555 [example, color, height, width, channel] tensor and argmaxxing over the color dimension, we find that 556 the grids are noisy and will often have the wrong colors for several random pixels. We developed 557 several methods for removing this noise: 532 Based on rules 1 and 3, we try to predict the shape of held-out outputs, prioritizing rule 1 over rule 533 3. If either rule holds, we force the postprocessing step to only consider the predicted shape by 534 overwriting the masks produced by the linear head layer. If neither rule holds, we make a temporary 535 prediction of the largest width and height out of the grids in the given ARC-AGI puzzle, and we allow 536 the masks to predict shapes that are smaller than that. 561 3. Construct an exponential moving average of the output color probabilities after taking the softmax. 562 Also construct an exponential moving average of the masks. 563 When applying these techniques, we always take the slice of highest probability given the mask, and 564 then we take the colors of highest probability afterwards. 565 We explored several different rules for when to select which method, and arrived at a combination of 566 1 and 2 with a few modifications: 567 - At every iteration, count up the sampled answer, as well as the exponential moving average answer 568 (decay = 0.97).

- If before 150 iterations of training, then downweight the answer by a factor of e
−10 569 . (Effectively, 570 don't count the answer.) 571 - If the answer is from the exponential moving average as opposed to the sample, then downweight the answer by a factor of e
−4 572 .

- Downweight the answer by a factor of e
−10∗uncertainty 573 , where uncertainty is the average (across 574 pixels) negative log probability assigned to the top color of every pixel.

575 **G Empirically Observed Abilities and Disabilities of CompressARC**

(b) CompressARC's solution to puzzle 28e73c20
(a) Puzzle 28e73c20 Figure 11: Puzzle 28e73c20, and CompressARC's solution to it.

576 A short list of abilities that can be performed by CompressARC includes: 577 - Assigning individual colors to individual procedures (see puzzle 0ca9ddb6) 578 - Infilling (see puzzle 0dfd9992) 579 - Cropping (see puzzle 1c786137) 580 - Connecting dots with lines, including 45 degree diagonal lines (see puzzle 1f876c06) 581 - Same color detection (see puzzle 1f876c06) 582 - Identifying pixel adjacencies (see puzzle 42a50994) 583 - Assigning individual colors to individual examples (see puzzle 3bd67248) 584 - Identifying parts of a shape (see puzzle 025d127b) 585 - Translation by short distances (see puzzle 025d127b)