# Related-work RAG improvements (branch `rag-improve-test`)

## Why

The earlier RAG runs (condition 7) showed that reviewers rarely used the retrieved works. A
reviewer named a shown work in 1 of 120 reviews, and novelty scores did not move. Three
problems explain this:

- the prompt said to use the evidence only "when it is relevant", and nothing checked whether it did;
- the RAG block sat after the full paper, in a long first message;
- 9 of the 60 packages contained the submission itself, and many shown works were already cited
  by the manuscript, so there was little new for reviewers to report.

All options are **off by default**. With the default config, conditions 1–7 and the web app run
exactly as before.

## What changed, by option

| Option | What it does | Main files |
|---|---|---|
| 1. Engaged prompt | Gives shown works stable labels [R1]..[Rk] and shows abstracts and cited status. Reviewers must fill `novelty_check` (one entry per claimed contribution, naming the closest shown work or "none", with the overlap level) and `missing_prior_work`, which may list only uncited works. Wrong labels trigger the normal schema retry. A short reminder is added in rebuttal rounds. | `prompts/reviewer_rag.py`, `rag/prompt_views.py`, `review_schema.py` (`validate_rag_review_fields`), `agents.py` (`extra_output_validator`), `mas_loop.py` |
| 2. Better retrieval | Adds one search query per claimed contribution (`claim_queries`). Reference expansion (`reference_expansion`) adds papers that cite the manuscript's most topical references, via Semantic Scholar; set `SEMANTIC_SCHOLAR_API_KEY` to avoid rate limits. `cutoff_mode` is `venue` (fixed date, used for evaluation) or `today` (for authors before submission). | `rag/related_work.py`, `rag/reference_expansion.py`, `rag/config.py` |
| 3. Cleaner evidence | Strict self-exclusion removes near-identical titles or the same arXiv ID/DOI. Each retrieved work is tagged as cited or not cited in the manuscript's reference list; cited works are kept and labeled, not removed. A relevance floor (`min_relevance`) applies. At most `max_shown` works are shown, with `min_uncited_shown` slots reserved for uncited works. This runs at run time, so old caches are cleaned too. | `rag/refine.py` |
| 4. Related-work analyst + author report | An optional analyst agent (`related_work_analyst`) reads the abstracts first and writes a short overlap report that goes to the reviewer. A "Possibly Missing References" report merges retrieval, the analyst and reviewers; it covers uncited, non-self works only, ranked by how many sources flag each work. It appears as a panel in the web app (on by default there). | `prompts/related_work_analyst.py`, `agents.py` (`RelatedWorkAnalyst`), `rag/missing_references.py`, `webapp/templates/index.html`, `webapp/static/style.css`, `config.py` |
| 5. Evaluation | Two new conditions: C8 (improved RAG) and C9 (C8 plus the analyst). Both use 1 iteration and 1 neutral reviewer, so they pair with C1. Adds `--provider` to `experiment.py`, a package builder, the first-public-date subset and an evaluation script. | `eval/experiment.py`, `eval/build_rag_packages.py`, `eval/first_public_dates.py`, `eval/rag_eval.py`, `eval/check_rag_usage.py` |

The C8/C9 run-time settings (`RAG_OVERRIDE_PRESETS` in `eval/experiment.py`) are:

- `prompt_style=engaged`
- `strict_self_exclusion`
- `tag_cited`
- `min_relevance=0.5`
- `max_shown=5`
- `min_uncited_shown=3`
- `missing_references_report`
- C9 also sets `related_work_analyst`.

## How to run (from the repository root)

Set your keys in the environment, not on the command line history if you can avoid it:

```bash
export ANTHROPIC_API_KEY=...            # or the key for the provider you use
export SEMANTIC_SCHOLAR_API_KEY=...     # optional, but reference expansion is rate-limited without it
```

**1. Build frozen packages with the improved retrieval.** This needs network access. It resumes
if interrupted.

```bash
python eval/build_rag_packages.py \
  --dataset eval/openreview_60_module_test.json \
  --output  eval/related_work_rag_60_v2.json \
  --provider claude --model claude-haiku-5-5
```

The default `--cutoff_mode venue --cutoff_date 2024-12-31` matches the earlier packages, so the
comparison stays fair. Add `--no_reference_expansion` if Semantic Scholar keeps refusing requests.

**2. Run C1 (no RAG), C2 (the old RAG prompt), C8 and C9 on the same papers with the same model.** C2 reads the same new packages with the old prompt, so C8 − C2 isolates the prompt and the evidence-cleaning changes.

```bash
python eval/experiment.py \
  --json_file eval/openreview_60_module_test.json \
  --conditions 1,2,8,9 \
  --provider claude --model claude-haiku-5-5 \
  --api_key "$ANTHROPIC_API_KEY" \
  --rag_cache eval/related_work_rag_60_v2.json \
  --output_dir eval/exp_results_rag_improve
```

C8 and C9 also work with the old `eval/related_work_rag_60.json`: the refinement step removes the
self-retrievals at run time, but the option 2 retrieval changes are then missing.

**3. Find papers newer than the model's training cutoff** (optional, needs network). Pass the
cutoff from the provider's official model documentation for the exact model you ran:

```bash
python eval/first_public_dates.py \
  --dataset eval/openreview_60_module_test.json \
  --model_cutoff YYYY-MM-DD \
  --output eval/first_public_dates_60.json
```

The script reports how many papers were first public, on arXiv or OpenReview, after the cutoff.
For older models this can be most or all of the 60; for recent models it can be zero.

**4. Evaluate.**

```bash
python eval/rag_eval.py \
  --dataset eval/openreview_60_module_test.json \
  --summary eval/exp_results_rag_improve/experiment_summary_<timestamp>.json \
  --control_cond 1 --rag_conds 2 8 9
# post-cutoff subset only:
python eval/rag_eval.py ... --subset eval/first_public_dates_60.json --output eval/rag_eval_post_cutoff.json
```

The report covers four groups of measures:

- **usage:** novelty checks that name a shown work, broken down by cited vs uncited works, and free-text mentions compared with C1;
- **missing-prior-work detection against human reviewers:** a keyword heuristic for related-work weaknesses, precision/recall, exact title or author-year hits, and statement similarity (MiniLM if installed);
- **citation validity:** invalid labels, unknown [R#] labels, and flagged works that are cited or are the submission itself; all should be 0;
- **paired C8/C9 vs C1:** accuracy, Spearman ρ, novelty score shift and decision agreement, with a 95% paper bootstrap.

## Cost

Each condition makes one reviewer call (plus a retry when a reply fails validation) and one conference call per paper. C9 adds one analyst
call. Package building adds about 3–5 LLM calls per paper (queries, claim queries, rerank).
For 60 papers that is roughly 60 × (4 conditions × 2 + 1 + 4) ≈ 800 calls, mostly paper-length inputs (about
20–30k tokens each). Check current prices for the model you choose before running.

## Tests

```bash
python -m unittest tests.test_rag_improvements
```

The tests run offline with mock clients and providers.
