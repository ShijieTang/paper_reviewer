# Paper Reviewer

A multi-agent system that simulates peer review of ML papers using LLM-based reviewer personas, author rebuttals, and conference recommendations.

---

## Project Structure

```
paper_reviewer/
├── mas_loop.py            # Main multi-agent review loop
├── agents.py              # Reviewer, Author, AIDetector, ConferenceRecommender agents
├── config.py              # Valid research topics
├── doc_preprocess.py      # PDF → Markdown conversion
├── modular_seg.py         # Markdown segmentation and reference normalization
├── check_citations.py     # Citation parsing utilities
├── requirements.txt
├── prompts/               # Persona prompts for each agent type
├── eval/                  # Experiment scripts and evaluation
│   ├── papers.json        # Ground truth paper metadata (24 papers)
│   ├── experiment.py      # Seven module-ablation conditions, parallel paper runs
│   ├── experiment_persona.py     # ABC × 3 iter experiment (agenttype=ABC)
│   ├── experiment_nopersona.py   # NNN × 3 iter baseline (agenttype=NNN)
│   ├── evaluation.py      # Benchmarking against OpenReview ground truth
│   ├── SRC.py             # Semantic Relevance & Confidence metric
│   └── experiment_trigger.py # Frozen T0/T1/T2 RAG screening
├── experiment_artifacts/ # Local-only results, snapshots, labels, and Drive handoff (gitignored)
├── webapp/                # Flask web interface
│   ├── app.py
│   ├── templates/
│   └── static/
└── data/
    ├── pdf/               # Input PDFs
    └── md/                # Converted Markdown files
```

---

## Setup

```bash
conda activate llm-project
pip install -r requirements.txt
```

Set your API key:
```bash
export API_KEY="your-api-key-here"
```

---

## Reviewer Personas

| Type | Persona | Focus |
|------|---------|-------|
| `reviewer_a` (A) | Ambitious researcher | Novelty & significance |
| `reviewer_b` (B) | Rigorous academic | Methodological soundness |
| `reviewer_c` (C) | Practitioner | Real-world applicability |
| `reviewer_nopersona` (N) | Neutral reviewer | No persona bias |

---

## Usage

### 1. Single paper review (CLI)

```bash
python mas_loop.py \
    --paper data/md/example_paper.md \
    --topic "Deep Learning" \
    --n_iter 3 \
    --output experiment_artifacts/local/results/my_review.txt
```

Arguments:
- `--paper` — path to `.md` file (default: `data/md/example_paper.md`)
- `--topic` — research area for persona injection (default: `""`)
- `--n_iter` — number of author-rebuttal iterations (default: `10`)
- `--output` — optional output file path

### 2. Web interface

```bash
python webapp/app.py
```

Open [http://localhost:5001](http://localhost:5001) in your browser. Upload a PDF, select reviewers, and stream results interactively.

---

## Experiments

All experiment scripts are run from the **project root**. Experiment artifacts
are stored under `experiment_artifacts/`, which is excluded by `.gitignore`.
The local `experiment_artifacts/README.md` documents the result inventory,
historical limitations, and shared Google Drive handoff. Do not use `git add -f`
on that directory. Existing GitHub history still contains previously committed
artifacts; local migration does not rewrite published history.

For the final T0/T1/T2 screening, see
[the execution and failure-handling protocol](docs/experiments/trigger_screening_protocol.md).

The archive utility previews its exact plan by default:

```bash
python scripts/archive_experiment_artifacts.py
```

Legacy runners reuse existing outputs. New trigger runs use strict provenance
checks and seal failed trajectories without drawing replacement samples.

### Module-ablation experiments (conditions 1-7)

The current `eval/experiment.py` preserves the remote module-ablation runner,
including parallel paper execution and precomputed RAG-package reuse. Its numeric
conditions are distinct from the historical C1-C5 conditions in
`eval/experiment_advanced.py` and the T0-T2 screening in `eval/experiment_trigger.py`.

| Condition | RAG | Rounds | Reviewers | Author rebuttal | Style evaluator |
|---|---|---|---|---|---|
| 1 | No | 1 | 1 neutral | Not applicable | No |
| 2 | Yes | 1 | 1 neutral | Not applicable | No |
| 3 | No | 2 | 1 neutral | No | Yes |
| 4 | No | 3 | 1 neutral | Yes | No |
| 5 | No | 1 | 3 personas | Not applicable | No |
| 6 | No | 2 | 1 neutral | No | No |
| 7 | Yes | 3 | 1 neutral | Yes | No |

```bash
python eval/experiment.py \
    --json_file eval/openreview_60_module_test.json \
    --api_key YOUR_API_KEY \
    --output_dir experiment_artifacts/local/eval/exp_results \
    --conditions 1,2,3,4,5,6,7 --concurrency 5
```

Output filenames:
```
{timestamp}_nagent=1_niter=1_paper={name}_cond=1_no_rag_1iter_1rev.txt
experiment_summary_{timestamp}.json
```

Pipeline outputs retain the remote `iterations` trace and also include
`workflow_status`, `turn_outcomes`, and `turn_failures`. Required role outputs use
shared validation in `review_schema.py`; exhausted failures stop the workflow.
Use `--disable_author_rebuttal` for intentional author ablations. Such runs are
not eligible for the T0-T2 protocol, which requires author turns.

### 3 persona reviewers × 3 iterations (agenttype=ABC)

```bash
python eval/experiment_persona.py \
    --api_key YOUR_API_KEY \
    --output_dir experiment_artifacts/local/eval/exp_results \
    [--md_dir data/md] \
    [--paper_id iclr_accept_001]
```

Output filenames:
```
paper={name}_niter=3_nagent=3_agenttype=ABC.txt
experiment_persona_summary_{timestamp}.json
```

### 3 no-persona reviewers × 3 iterations (agenttype=NNN)

```bash
python eval/experiment_nopersona.py \
    --api_key YOUR_API_KEY \
    --output_dir experiment_artifacts/local/eval/exp_results \
    [--md_dir data/md] \
    [--paper_id iclr_accept_001]
```

Output filenames:
```
paper={name}_niter=3_nagent=3_agenttype=NNN.txt
experiment_nopersona_summary_{timestamp}.json
```

---

## Evaluation

Compare experiment results against OpenReview ground truth (SRC metric + accept/reject accuracy):

```bash
python eval/evaluation.py \
    --papers eval/papers.json \
    --openreviewer experiment_artifacts/local/eval/openreviewer.json \
    --paperreviewer experiment_artifacts/local/eval/paperreviewer.json \
    --exp_summary experiment_artifacts/local/eval/exp_results/experiment_summary_{timestamp}.json \
    --nopersona_summary experiment_artifacts/local/eval/exp_results/experiment_nopersona_summary_{timestamp}.json \
    --output_dir experiment_artifacts/local/eval/eval_results
```

Key arguments:
- `--exp_summary` — path to an experiment summary (A/B, C1–C5, or T0–T2)
- `--baseline_summary` — path to a single-condition baseline summary
- `--nopersona_summary` — path to a no-persona summary
- `--output_file` — exact evaluation output path
- `--paper_ids` — space-separated subset of papers to evaluate

---

## NLPeer ARR-22 Benchmark Conversion

`scripts/convert_nlpeer_arr22.py` converts NLPeer ARR-22 into one JSONL record per paper with a simple schema:

```json
{"paper_id": "arr22_xxx", "accept_or_not": "accept", "score": 3, "reviews": [{"reviewer_id": "review_1", "strengths": ["..."], "weaknesses": ["..."]}]}
```

ARR-22 is used for the first NLPeer conversion because its review objects expose structured `report` fields and `scores`, including strength-like fields, weakness-like fields, and overall scores. The converter does not use an LLM and does not split free-form review text. It exports only reviews where strengths, weaknesses, and an overall score are available from structured fields, then averages the valid review scores for the paper-level `score`.

Important label warning: NLPeer ARR-22 contains papers later accepted at ACL/NAACL, so `accept_or_not` is written as constant `"accept"`. This label is useful for compatibility with existing benchmark readers, but it is not suitable for accept/reject prediction or balanced decision-label evaluation.

Install the official NLPeer package and run:

```bash
pip install git+https://github.com/UKPLab/nlpeer

python scripts/convert_nlpeer_arr22.py \
    --nlpeer-root /path/to/NLPeer \
    --out eval/nlpeer_arr22.jsonl \
    --stats-out eval/nlpeer_arr22_stats.json
```

Optional arguments:
- `--dataset` resolves aliases such as `ARR-22` or `ARR22` against `nlpeer.DATASETS`.
- `--version` selects the NLPeer paper version, defaulting to `1`.

The stats JSON includes exported/skipped paper and review counts, score distributions, field-name distributions observed in the local dataset, and the constant-label warning.

---

## OpenReviewer Baseline (via HuggingFace Spaces)

To run the OpenReviewer baseline on a GPU (recommended: Google Colab with GPU):

```bash
git clone https://huggingface.co/spaces/maxidl/openreviewer
cd openreviewer
pip install -r requirements.txt
pip install spaces gradio huggingface_hub
```

Edit line 256 of `app.py`:
```python
demo.launch(share=True)
```

Then run:
```bash
python app.py
```

Copy the public URL into your browser. (Keep colab terminal running when using the public url)
