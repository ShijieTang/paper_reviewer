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
│   ├── experiment.py      # Condition A (single) vs B (multi-agent ABC) experiment
│   ├── experiment_persona.py     # ABC × 3 iter experiment (agenttype=ABC)
│   ├── experiment_nopersona.py   # NNN × 3 iter baseline (agenttype=NNN)
│   ├── evaluation.py      # Benchmarking against OpenReview ground truth
│   ├── SRC.py             # Semantic Relevance & Confidence metric
│   ├── exp_results/       # Per-paper result files (.txt JSON)
│   ├── exp_baseline_results/
│   └── eval_results/      # Evaluation output JSONs
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
    --output results/my_review.txt
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

All experiment scripts are run from the **project root**. Results auto-skip papers that already have output files.

### Condition A vs B (single-agent vs multi-agent)

Runs two conditions on all 24 papers in `eval/papers.json`:
- **Condition A**: `reviewer_a` × 1 iteration
- **Condition B**: `reviewer_a, reviewer_b, reviewer_c` × 3 iterations

```bash
python eval/experiment.py \
    --json_file eval/papers.json \
    --api_key YOUR_API_KEY \
    --output_dir eval/exp_results \
    [--paper_id iclr_accept_001]
```

Output filenames:
```
{timestamp}_nagent=1_niter=1_paper={name}_cond=A_single.txt
{timestamp}_nagent=3_niter=3_paper={name}_cond=B_multi.txt
experiment_summary_{timestamp}.json
```

### 3 persona reviewers × 3 iterations (agenttype=ABC)

```bash
python eval/experiment_persona.py \
    --api_key YOUR_API_KEY \
    --output_dir eval/exp_results \
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
    --output_dir eval/exp_results \
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
    --openreviewer eval/openreviewer.json \
    --paperreviewer eval/paperreviewer.json \
    --exp_summary eval/exp_results/experiment_summary_{timestamp}.json \
    --baseline_summary eval/exp_results/experiment_nopersona_summary_{timestamp}.json \
    --output_dir eval/eval_results
```

Key arguments:
- `--exp_summary` — path to `experiment_summary_*.json` (Condition A & B)
- `--baseline_summary` — path to `experiment_nopersona_summary_*.json` or `experiment_persona_summary_*.json`
- `--our_results` — path to a single result `.txt` file
- `--conf_threshold` — accept/reject score threshold (default: `6.0`)
- `--paper_ids` — space-separated subset of papers to evaluate

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