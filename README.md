# AML/Fraud Triage Assistant

An AML/fraud triage system over the PaySim synthetic mobile-money dataset, built in two layers: a
gradient-boosted classifier that scores a transaction for fraud risk, and a retrieval-grounded LLM
agent that turns a flagged transaction into a structured escalate/monitor/close recommendation with a
cited rationale. An eval harness measures both.

## Results

Phase 1 classifier, from `reports/phase1_report.json`:

| Metric | Value |
| --- | --- |
| Precision | 0.906 |
| Recall | 0.756 |
| PR-AUC | 0.884 |
| Operating threshold | 0.571 |

At a score cutoff of 0.571, roughly 9 in 10 flagged transactions are genuinely fraud, and roughly 3
in 4 of the fraud present is caught. The PR-AUC of 0.884 summarises the precision/recall trade-off
across all thresholds, not just the chosen one. Accuracy is not reported: at PaySim's fraud rate —
well under 1% — a model that always predicts "not fraud" scores above 99% while catching nothing.
The numbers come from a time-based split, trained on early simulation steps and evaluated on later
ones, so no future information reached training.

The Phase 3 triage agreement rate is **not yet measured**. The `report()` function that computes it
exists and is tested, but the live run that populates `reports/phase3_triage_eval.json` requires an
`ANTHROPIC_API_KEY` and has not been executed. No number is estimated here in its place. When run,
that report will contain `decision_agreement_rate`, `citation_present_rate`, and
`judge_agreement_rate` over a 16-case hand-labeled eval set.

## Method

### Phase 1 — the classifier

- A PaySim sample of 200,000 rows (`data/paysim_sample.csv`), drawn with a fixed seed.
- A time-based split on PaySim's `step` column at `split_step=355`, chosen over a random split
  because a random shuffle lets the model learn from transactions that occur after the ones it is
  tested on.
- Feature engineering drops the account identifiers and the label-adjacent `isFlaggedFraud`, derives
  origin and destination balance deltas, and one-hot encodes transfer and cash-out.
- Class imbalance is handled with `scale_pos_weight = negatives / positives`, computed on the
  training split only.
- XGBoost (`n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="aucpr", random_state=42`),
  chosen because the features are already a small numeric table — there is no representation for a
  neural net to learn — and because boosted trees stay inspectable for a compliance reviewer.
- The operating threshold is selected by maximising recall subject to precision >= 0.90.

### Phase 2 — the triage agent

- A six-document typology corpus (`data/typologies.json`).
- Hybrid retrieval blending TF-IDF cosine similarity with `all-MiniLM-L6-v2` sentence embeddings as
  `alpha * tfidf + (1 - alpha) * embedding`, because keyword matching misses paraphrase and
  embeddings drift toward the merely topical.
- A forced Anthropic tool call (`submit_triage_decision`) constrains the output to a decision enum, a
  rationale, and a list of cited typology ids.
- A deterministic parser rejects any cited id the model was not shown on that call, so a fabricated
  or unshown citation raises rather than reaching a reviewer.

The agent drafts a recommendation for a human to review. It never makes an automatic decision.

### Phase 3 — the eval harness

- A hand-labeled golden set of 16 cases (`data/triage_eval_set.jsonl`), each carrying a
  `label_decision` and a one-sentence note written by hand rather than generated.
- `deterministic_score` checks decision match and citation presence.
- `llm_judge_score` asks a model whether the rationale plausibly follows from the typology text it
  cited.
- `report` aggregates the three rates.

They stay three separate rates rather than one blended score because each points somewhere
different: a low decision rate points at the classifier or the decision prompt, a low citation rate
at tool use, a low judge rate at grounding. Averaging them lets a good rate conceal a bad one.

## Running it

From the repository root:

```sh
uv sync
uv run python -m aml_triage.scripts.train_baseline
uv run pytest
```

`data/` is gitignored — the PaySim CSV is large and derived — so the sample must be present before
the training script runs. The Phase 3 live run additionally needs `ANTHROPIC_API_KEY` exported.

## Layout

- `data.py` — loading and dtype coercion
- `split.py` — the time-based split
- `features.py` — feature engineering
- `imbalance.py` — the class weight
- `model.py` — training
- `evaluate.py` — classifier metrics and threshold selection
- `retrieval.py` — typology retrieval
- `triage_schema.py` — the tool schema and citation guard
- `triage.py` — the end-to-end agent
- `eval.py` — the triage eval harness
- `scripts/train_baseline.py` — the Phase 1 pipeline runner

## What this deliberately simplifies

- PaySim ships an `isFraud` column. Real AML has no ground-truth labels — what institutions have is
  suspicious-activity reports, an analyst's suspicion, rarely followed by any word on what actually
  happened.
- Fraud is not money laundering. Fraud has a victim who complains, which is why labels for it exist
  at all; laundering has no complainant.
- The failure mode optimised against here is missed fraud. In production the harder problem is
  usually false-positive volume — alerts that bury the analysts who must clear them.
- There is no network layer. Mule layering and round-tripping are patterns *between* accounts; here
  they are approximated from a single transaction's balance columns. Real systems add graph structure
  and entity resolution.
- Retrieval over six typology documents is below the scale at which retrieval earns its keep; the
  mechanism is the point, not the corpus size.
- 16 hand-labeled cases is a floor that proves the pipeline runs end to end, not a sample size that
  supports a strong claim about agent quality. A real model-validation document would stratify dozens
  or hundreds of cases by decision type and transaction family and have them reviewed by people who
  know the typologies.
