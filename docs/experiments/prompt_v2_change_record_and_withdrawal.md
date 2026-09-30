# Prompt-v2 change record and withdrawal

Date: 2026-08-04

Status: Withdrawn from the active RAG design
Rollback target: the pre-v2 detailed RAG view without update-message duplication

## Decision

The Prompt-v2 RAG presentation is withdrawn from the active implementation. Its
experiment artifacts remain unchanged so the result can still be reported as an
ablation and reproduced from the archived summaries.

This is a selective rollback. It restores the detailed RAG evidence formatter
and removes Prompt-v2's repeated update-round injection without removing later
JSON validation, retry, repair, or evaluation fixes.

## What “Prompt-v2” can be verified to mean

The experiment summary does not contain a prompt-version field, prompt hash,
Git commit, or source snapshot. The `promptv2-rep2` name is a directory label,
not recorded experiment metadata. Exact provenance was recovered instead from
the local Codex implementation log:

`~/.codex/sessions/2026/07/28/rollout-2026-07-28T22-51-37-019fa935-b6a3-7863-af52-735ab0f37acb.jsonl`

The approval is recorded at line 2164, the applied three-file patch at lines
2197–2229, verification at lines 2280–2295, and the implementation handoff at
line 2303. The patch was applied at 2026-07-30 15:08:08Z
(2026-07-30 23:08:08 +0800).

The following facts are verified:

- The Prompt-v2 Replicate 2 source run is `2607311842`; its repaired successor
  is `260801230040`.
- It used provider `openrouter` and model
  `deepseek/deepseek-chat-v3.1` on the same 24 papers and the same C1–C5
  condition definitions as the V31 and V31 Fresh experiments.
- The committed reviewer, reviewer-iteration, and author prompt files are
  byte-identical between the repository state preceding these experiments and
  the current `HEAD`:
  - `prompts/reviewer_common.py`
  - `prompts/reviewer_iter.py`
  - `prompts/author.py`
- Prompt-v2 was an explicitly approved three-file change:
  - `rag/prompt_views.py` changed the detailed evidence block to a compact,
    summary-only view;
  - `mas_loop.py` inserted that identical block into every later reviewer
    iteration immediately before `###TASK###`;
  - `tests/test_related_work_rag.py` was updated to enforce both behaviors.
- The V31 source summary `2607300838` was completed at 2026-07-30 11:42:48,
  before that formatter change.
- V31 Fresh began at 2026-07-30 23:16:15, approximately eight minutes after the
  Prompt-v2 patch, and Prompt-v2 Replicate 2 began at 2026-07-31 18:42:14.
  Therefore V31 Fresh and Prompt-v2 Replicate 2 both used Prompt-v2 behavior.

No reviewer persona, reviewer-iteration template, author template, RAG
construction, cache, or experiment-condition change was part of Prompt-v2.

## Exact RAG prompt-view change

The underlying RAG package structure and retrieval taxonomy did not change.
All compared summaries retain the same package fields and the same six query
groups:

1. `same_problem`
2. `same_method`
3. `same_constraints`
4. `benchmark_baseline`
5. `novelty_competitor`
6. `limitations_counterevidence`

What changed was the subset of the completed RAG package shown to reviewers.

### Pre-v2 detailed view

The formatter exposed:

- the synthesized related-work summary;
- up to eight reranked related papers;
- paper title, year/date, sources, and authors;
- reranker relevance score and rationale;
- auxiliary OpenReview review-memory calibration when available, including the
  selected related paper, decision pattern, score range, common strengths,
  common weaknesses, and calibration notes;
- retrieval cutoff statistics.

The `max_papers` parameter actively limited the paper list. A non-empty package
could still produce a block even when `related_work_summary` was empty.

### Prompt-v2 compact view

The formatter was changed to expose only:

- `###RAG_EVIDENCE###`;
- the synthesized `related_work_summary`;
- a short evidence-use boundary saying that the submitted paper remains the
  primary evidence;
- `###END_RAG_EVIDENCE###`.

It additionally:

- removed `_paper_lookup`;
- ignored `max_papers`;
- omitted reranked paper metadata and rationales;
- omitted review-memory calibration;
- omitted cutoff statistics;
- returned an empty string whenever the synthesized summary was empty, even if
  other package evidence existed.

The full RAG package continued to be stored in experiment results for audit.

### Prompt-v2 iteration placement

Before Prompt-v2, the formatted RAG block was appended to `reviewer_paper` when
the reviewer agent was initialized, so it remained available in the reviewer's
system context. Later reviewer-update user messages contained the author
response, optional style analysis, and `###TASK###`, without another copy of the
RAG block.

Prompt-v2 changed `construct_reviewer_prompt` to accept `rag_prompt_block` and
append the same immutable block immediately before `###TASK###` in every later
reviewer iteration. It also changed the initial instruction wording from
“related-work block” to “related-work summary.” The initial reviewer still
received the block through `reviewer_paper`; it was not duplicated in the
initial user message.

## Changes explicitly outside this rollback

The following changes occurred later or serve correctness rather than the
Prompt-v2 RAG presentation, so they are retained:

- JSON-object validation and retry handling in `agents.py`;
- omission of invalid structured returns instead of storing parse-error records;
- repaired experiment summaries and raw replacement results;
- evaluation support for advanced C1–C5 summaries.

## RAG-content confounder

The formatter was not the only RAG difference between the runs. The stored
package IDs are content-derived from each paper's generated retrieval queries,
and none of the 24 per-paper package IDs match between V31, V31 Fresh, and
Prompt-v2. Retrieval completeness also differed:

| Source run | Papers with RAG warnings | Mean retrieved items used | Papers with zero evidence |
|---|---:|---:|---:|
| V31 `2607300838` | 10/24 | 11.125 | 0 |
| V31 Fresh `2607302348` | 15/24 | 7.750 | 2 |
| Prompt-v2 `2607311842` | 13/24 | 9.333 | 1 |

Thus the observed metric differences cannot be assigned solely to the prompt
formatter. Future prompt comparisons must reuse identical precomputed packages.

## Experimental outcome motivating withdrawal

C3 and C5 are the closest RAG ablation: both use three neutral reviewers and
three iterations, while C3 enables RAG and C5 does not.

| Repaired run | C3 SRC overall | C5 SRC overall | C3 − C5 | Decision-accuracy change |
|---|---:|---:|---:|---:|
| V31 Fresh `260802231042` | 0.4234 | 0.4254 | -0.0020 | +20.83 pp |
| Prompt-v2 Replicate 2 `260801230040` | 0.4222 | 0.4326 | -0.0104 | -4.16 pp |
| V31 `260804084236` | 0.4255 | 0.4328 | -0.0073 | -4.17 pp |
| DeepSeek `260803233627` | 0.4399 | 0.4410 | -0.0011 | +4.17 pp |

The current evidence supports withdrawing Prompt-v2 as the active design
because it did not yield a reliable RAG benefit and produced the largest C3 SRC
penalty. It does not prove that the compact formatter alone caused the result:
V31 Fresh used the same formatter, RAG packages differed between runs, model
generation is stochastic, and repaired summaries contain different replacement
subsets.

## Withdrawal implementation

The rollback restores the committed detailed implementation of
`format_rag_prompt_block` in `rag/prompt_views.py`, including the paper list,
review-memory context, and cutoff report. It also restores the pre-v2
`construct_reviewer_prompt` interface and stops duplicating RAG evidence in
later reviewer-update messages.

The associated focused tests are restored to verify that RAG is formatted in
the detailed form and injected through `reviewer_paper`, not repeated in every
update message.

No experiment result, cache, evaluation report, or raw model output is deleted
or modified.

## Withdrawal verification

- `rag/prompt_views.py` is byte-identical to the committed pre-v2 formatter
  (`HEAD` blob `3acb1deb8ae0a78c5792d3b4641025ca3b95122b`).
- `mas_loop.py` no longer accepts or appends `rag_prompt_block` in
  `construct_reviewer_prompt`; its remaining working-tree changes are the later
  structured-JSON correctness fixes.
- `tests/test_related_work_rag.py` is restored to its pre-v2 behavior checks.
- Focused verification command:

  ```bash
  python -m pytest -q \
    tests/test_related_work_rag.py \
    tests/test_reviewer_prompt_contract.py \
    tests/test_ai_detector_optional.py
  ```

  Result: `38 passed`.

The repository-wide suite has unrelated existing dependency and test-order
failures, so the focused affected-area suite is the rollback acceptance check.

## Recommended next experiment

Do not immediately treat the restored detailed formatter as the final RAG
design. Use it as a documented baseline for a controlled comparison with a new
gated design:

- fixed model and provider;
- fixed paper set;
- fixed precomputed RAG packages;
- identical reviewer and author prompts;
- identical JSON retry behavior;
- C5 no-RAG control;
- detailed baseline, compact withdrawn version, and a new gated/source-grounded
  version;
- multiple independent repetitions;
- paired per-paper analysis of SRC, decision accuracy, score correlation, and
  citation/evidence quality.
