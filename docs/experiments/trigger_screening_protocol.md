# Final RAG Screening: Execution and Failure Handling

This is an operational reference for the prepared protocol, not a new experimental result. For the single summary of code changes, C1-C5 definitions, completed experiments, and conclusions, see the local [experiment archive README](../../experiment_artifacts/README.md). That archive is intentionally excluded from Git.

The commands below have not been executed as a new paid experiment.

## Comparison

- T0 uses the C5 setup: no RAG, three neutral reviewers, three review rounds, with author rebuttals.
- T1 uses the C3 setup with an immutable related-work package.
- T2 is a derived policy, not another independent model call: select T1 for a gate-PASS package, otherwise T0.
- Use 36 fixed validation papers and three repetitions. The analysis unit is the paper, not an individual call.
- T2 minus T0 is the primary comparison. Do not switch to T1 after inspecting outcomes.
- Screening thresholds use point estimates. The 80% paired, stratified bootstrap intervals describe uncertainty; they do not establish formal statistical non-inferiority or a confirmatory full-dataset result.

## Reliability rules

Reviewer generation, audit, repair, and evaluation share the same required score fields and output validation. Author responses must address the intended reviewer. A workflow is complete only when every scheduled required turn succeeds.

The integrated pipeline retains the remote per-round `iterations` trace alongside
the new turn-level diagnostics. JSON fences and trailing commas are accepted
without altering quoted text; semantic validation remains strict. Ordinary
module-ablation runs may disable author rebuttals, but T0-T2 completeness requires
them. Current related-work packages omit the removed review-memory subsystem;
legacy packages are accepted only when that subsystem is explicitly disabled.

Evaluate each sealed trigger summary separately. Multi-summary evaluation remains
available for ordinary legacy/module-ablation runs. These integration changes
alter code hashes: do not reuse previously frozen packages or started trials under
the new code, or rewrite their stored hashes to make them pass validation.

The agent applies a fixed budget of up to four attempts to the same request/context. Invalid JSON/schema output and qualifying transient errors can be retried; terminal provider errors stop the turn. Provider SDK retries may occur separately. Invalid replies are not added to subsequent conversation context.

For example, an empty second-round author response must not be hidden by valid third-round reviews. Exhausted retries produce a stored failure and stop the workflow. The trigger runner reserves a result before its first call, preserves interruption/failure records, and prevents automatic resampling of incomplete started arms.

Service failures mean incomplete execution, not evidence that RAG is effective or ineffective. Do not delete reservations or selectively rerun failures until the outputs look acceptable. Any protocol adjustment or replacement batch needs a separately documented decision.

Older experiments lack complete turn records; this protocol cannot retrospectively certify them. The legacy C1-C5 repair script must not repair sealed trigger runs.

## Storage and blinding

All new outputs default to the Git-ignored experiment_artifacts directory. Never force-add raw outputs, labels, or audit keys to Git.

The complete archive is for authorized project members. A blinded auditor receives only blinded_pairs.json and judgments.json, not private labels, source-arm outputs, or private_key.json. Keep the private key with the operator until judgments are sealed. Review content can still suggest arm identity; anonymous labels do not guarantee perfect blinding.

## Commands

Run from the repository root. Before paid execution, confirm provider/model availability and set OPENROUTER_API_KEY in the environment. Do not put credentials in command arguments, documentation, or the archive.

### 1. Prepare the RAG packages once

This step can incur retrieval and model costs.

~~~bash
python eval/experiment_trigger.py --prepare-rag-only \
  --provider openrouter --model deepseek/deepseek-chat-v3.1
~~~

### 2. Audit the frozen packages

This step does not call the model.

~~~bash
python eval/experiment_trigger.py --audit-rag-only \
  --provider openrouter --model deepseek/deepseek-chat-v3.1
~~~

Verify that all 36 packages are valid and gate-PASS coverage is 8-28 papers. Missing abstracts or traceable verification sources fail the gate. The final analysis also checks label support in each gate group. Do not rebuild packages, tune the gate, or change code after screening starts to improve the observed pass rate.

### 3. Run the three formal repetitions

This step incurs model costs.

~~~bash
for rep in 1 2 3; do
  python eval/experiment_trigger.py \
    --provider openrouter --model deepseek/deepseek-chat-v3.1 \
    --repeat_id "$rep" \
    --output_dir "experiment_artifacts/trigger/runs/rep$rep" || break
done
~~~

Inspect preserved files after any interruption or failure. Complete arms can be reused; started incomplete arms cannot be automatically rerun. Do not copy a failed run into another directory or use legacy repair to bypass sealing. Keep smoke tests in a separate directory and exclude them from formal repetitions.

### 4. Evaluate with the frozen scoring identity

~~~bash
for rep in 1 2 3; do
  python eval/evaluation.py \
    --papers eval/trigger_validation_36.evaluation.json \
    --exp_summary "experiment_artifacts/trigger/runs/rep$rep/experiment_trigger_summary_rep-$rep.json" \
    --output_file "experiment_artifacts/trigger/evaluation/rep$rep.json" || break
done
~~~

The immutable embedding revision is defined in eval/evaluation_protocol.py. First use may require downloading weights. Changed evaluator/SRC code is rejected for a sealed run: restore the original code rather than redefining the metric after seeing results.

### 5. Prepare, complete, and seal the blinded assessment

~~~bash
python eval/prepare_trigger_evidence_audit.py prepare \
  --summary 'experiment_artifacts/trigger/runs/rep*/experiment_trigger_summary_rep-*.json'
~~~

An auditor who does not know A/B identities or ground-truth labels checks the target paper and sources, then fills judgments.json. The operator retains private_key.json. Do not use a public random seed or overwrite the audit to obtain a new assignment.

After the human assessment is complete:

~~~bash
python eval/prepare_trigger_evidence_audit.py finalize
python eval/analyze_trigger_screening.py \
  --pair experiment_artifacts/trigger/runs/rep1/experiment_trigger_summary_rep-1.json experiment_artifacts/trigger/evaluation/rep1.json \
  --pair experiment_artifacts/trigger/runs/rep2/experiment_trigger_summary_rep-2.json experiment_artifacts/trigger/evaluation/rep2.json \
  --pair experiment_artifacts/trigger/runs/rep3/experiment_trigger_summary_rep-3.json experiment_artifacts/trigger/evaluation/rep3.json \
  --evidence-review experiment_artifacts/trigger/evidence_audit/evidence_review.json
~~~

Only GO_GATED_RAG_CONFIRMATORY_HOLDOUT supports proceeding to an unused confirmatory holdout. Previously inspected papers must not be relabeled as independent validation. NO_GO_ABANDON_RAG ends this RAG screening route. INVALID_EVIDENCE_REVIEW requires checking/restoring the sealed evidence, not collecting replacement judgments to obtain PASS.

The archived copy of this operational document remains the original historical snapshot; this working English document does not change its recorded hash or any experimental artifact.
