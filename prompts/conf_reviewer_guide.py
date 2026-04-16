iclr_2025_reviewer_guide = """
##Summary##
Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.

##Soundness##
Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence. Choose from the following:
4: excellent
3: good
2: fair
1: poor

##Presentation##
Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work. Choose from the following:
4: excellent
3: good
2: fair
1: poor

##Contribution##
Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader ICLR community? Choose from the following:
4: excellent
3: good
2: fair
1: poor

##Strengths##
A substantive assessment of the strengths of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage reviewers to be broad in their definitions of originality and significance. For example, originality may arise from a new definition or problem formulation, creative combinations of existing ideas, application to a new domain, or removing limitations from prior results.

##Weaknesses##
A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc.

##Questions##
Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This is important for a productive rebuttal and discussion phase with the authors.

##Flag For Ethics Review##
If there are ethical issues with this paper, please flag the paper for an ethics review and select area of expertise that would be most useful for the ethics reviewer to have. Please select all that apply. Choose from the following:
No ethics review needed.
Yes, Discrimination / bias / fairness concerns
Yes, Privacy, security and safety
Yes, Legal compliance (e.g., GDPR, copyright, terms of use)
Yes, Potentially harmful insights, methodologies and applications
Yes, Responsible research practice (e.g., human subjects, data release)
Yes, Research integrity issues (e.g., plagiarism, dual submission)
Yes, Unprofessional behaviors (e.g., unprofessional exchange between authors and reviewers)
Yes, Other reasons (please specify below)

##Details Of Ethics Concerns##
Please provide details of your concerns.

##Rating##
Please provide an "overall score" for this submission. Choose from the following:
1: strong reject
3: reject, not good enough
5: marginally below the acceptance threshold
6: marginally above the acceptance threshold
8: accept, good paper
10: strong accept, should be highlighted at the conference
"""


# https://icml.cc/Conferences/2025/ReviewerInstructions
icml_2025_reviewer_guide = """
##Summary##
Briefly summarize the paper (including the main findings, main results, main algorithmic/conceptual ideas, etc. that the paper claims to contribute). This summary should not be used to critique the paper. A well-written summary should not be disputed by the authors of the paper or other readers.

##Claims and Evidence##
1. Are the claims made in the submission supported by clear and convincing evidence? If not, which claims are problematic and why?
2. Do proposed methods and/or evaluation criteria (e.g., benchmark datasets) make sense for the problem or application at hand?
3. Did you check the correctness of any proofs for theoretical claims? Please specify which ones, and discuss any issues.
4. Did you check the soundness/validity of any experimental designs or analyses? Please specify which ones, and discuss any issues.
5. Did you review the supplementary material? Which parts?

##Relation to Prior Works##
1. How are the key contributions of the paper related to the broader scientific literature? Be specific in terms of prior related findings/results/ideas/etc.
2. Are there related works that are essential to understanding the (context for) key contributions of the paper, but are not currently cited/discussed in the paper? Be specific in terms of prior related findings/results/ideas/etc. (Example: “The key contribution is a linear-time algorithm, and only cites a quadratic-time algorithm for the same problem as prior work, but there was also an O(n log n) time algorithm for this problem discovered last year, namely Algorithm 3 from [ABC’24] published in ICML 2024.”)
3. How well-versed are you with the literature related to this paper? (Examples: “I keep up with the literature in this area.”; “I am only familiar with a few key papers in this area, namely [ABC’02], [DEF’04], and [GHI’05].”) Note: Your response to this item will not be visible to authors. Please also see instructions regarding concurrent work.

##Other Aspects##
1. Enter any comments on other strengths and weaknesses of the paper, such as those concerning originality, significance, and clarity. We encourage you to be open-minded in terms of potential strengths. For example, originality may arise from creative combinations of existing ideas, removing restrictive assumptions from prior theoretical results, or application to a real-world use case (particularly for application-driven ML papers, indicated in the flag above and described in the Reviewer Instructions).
2. If you have any other comments or suggestions (e.g., a list of typos), please write them here.

##Questions for Authors##
If you have any important questions for the authors, please carefully formulate them here. Please reserve your questions for cases where the response would likely change your evaluation of the paper, clarify a point in the paper that you found confusing, or address a critical limitation you identified. Please number your questions so authors can easily refer to them in the response, and explain how possible responses would change your evaluation of the paper.

##Ethical Issues##
1. If you believe there are ethical issues with this paper, please flag the paper for an ethics review. For guidance on when this is appropriate, please review the ethics guidelines.
2. If you flagged this paper for ethics review, what area of expertise would it be most useful for the ethics reviewer to have? Please click all that apply:
    2.1. Discrimination / Bias / Fairness Concerns
    2.2. Inappropriate Potential Applications & Impact  (e.g., human rights concerns)
    2.3. Privacy and Security
    2.4. Legal Compliance (e.g., GDPR, copyright, terms of use)
    2.5. Research Integrity Issues (e.g., plagiarism)
    2.6. Responsible Research Practice (e.g., IRB, documentation, research ethics)
    2.7. Other
3. If you flagged this paper for an ethics review, please explain your concerns in detail.

##Overall Recommendation##
Indicate an overall recommendation:
    5. Strong accept
    4. Accept
    3. Weak accept (i.e., leaning towards accept, but could also be rejected)
    2. Weak reject (i.e., leaning towards reject, but could also be accepted)
    1. Reject
"""

# https://neurips.cc/Conferences/2025/ReviewerGuidelines
neurips_2025_reviewer_guide = """
##Summary##
# Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary. This is also not the place to paste the abstract—please provide the summary in your own understanding after reading.

##Strengths and Weaknesses##
Please provide a thorough assessment of the strengths and weaknesses of the paper. A good mental framing for strengths and weaknesses is to think of reasons you might accept or reject the paper. Please touch on the following dimensions:
1. Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?
2. Clarity: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
3. Significance: Are the results impactful for the community? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance our understanding/knowledge on the topic in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?
4. Originality: Does the work provide new insights, deepen understanding, or highlight important properties of existing methods? Is it clear how this work differs from previous contributions, with relevant citations provided? Does the work introduce novel tasks or methods that advance the field? Does this work offer a novel combination of existing techniques, and is the reasoning behind this combination well-articulated? As the questions above indicates, originality does not necessarily require introducing an entirely new method. Rather, a work that provides novel insights by evaluating existing methods, or demonstrates improved efficiency, fairness, etc. is also equally valuable.

##Quality##
Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the quality of the work.
4 excellent
3 good
2 fair
1 poor

##Clarity##
Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the clarity of the paper.
4 excellent
3 good
2 fair
1 poor

##Significance##
Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the significance of the paper.
4 excellent
3 good
2 fair
1 poor

##Originality##
Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the originality of the paper.
4 excellent
3 good
2 fair
1 poor

##Questions##
Please list up and carefully describe questions and suggestions for the authors, which should focus on key points (ideally around 3–5) that are actionable with clear guidance. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. You are strongly encouraged to state the clear criteria under which your evaluation score could increase or decrease. This can be very important for a productive rebuttal and discussion phase with the authors.

##Limitations##
Have the authors adequately addressed the limitations and potential negative societal impact of their work? If so, simply leave “yes”; if not, please include constructive suggestions for improvement. In general, authors should be rewarded rather than punished for being up front about the limitations of their work and any potential negative societal impact. You are encouraged to think through whether any critical points are missing and provide these as feedback for the authors.

##Overall##
Please provide an "overall score" for this submission. Choices:
6: Strong Accept: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
1: Strong Reject: For instance, a paper with well-known results or unaddressed ethical considerations

##Confidence##
Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation.  Choices
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.
"""