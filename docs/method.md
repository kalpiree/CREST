# Method and experiment protocols

## Paper-to-code map

| Paper component | Implementation |
| --- | --- |
| RecAtoms, RRM serialization and whole-identity omissions | `src/crest/records.py` |
| Temporal influence, omission cost and beam search | `src/crest/screening.py` |
| Appended-endpoint order statistic and integer promotion budget | `src/crest/calibration.py` |
| Exact-K feasibility and minimum-rank-cost selection | `src/crest/selection.py` |
| Instruction injection and history insertion | `src/crest/rq1_attacks.py` |
| TextSimu rewriting adaptation | `src/crest/textsimu_attack.py`, `src/crest/rq1_rewriting_export.py` |
| Record-based defenses | `src/crest/defenses/` |
| Recall, NDCG and promotion | `src/crest/metrics.py` |
| Paper-facing aggregation and table generation | `src/crest/reporting.py` |
| Independent calibration blocks and control comparisons | `src/crest/controls.py` |
| Component ablation | `src/crest/ablation.py` |

## Recommendation inputs

Each decision contains a user ID, an ordered candidate list and typed records. The prompt keeps the user and candidate-label mapping outside `<recommendation_data>`. Each RecAtom becomes a `<record>` containing its identity, type and ordered fields. Record contents are escaped with Python's `html.escape(..., quote=True)`. Omitting an identity removes its complete record at every appearance and never removes its associated item from the candidate set.

The recommender returns exactly K distinct candidate labels as a JSON array, ordered by predicted relevance. Constrained decoding and validation reject invalid outputs; the runner does not fill missing recommendations. Every baseline and omission query uses the same frozen model, candidate order and output format. Spotlighting alone adds caret datamarking and its associated instruction.

## Sequence and attack construction

Dataset preprocessing retains stable source-row identities and applies chronological train, development, calibration and test boundaries. Each decision uses three preceding history records, one held-out positive, a preselected target, and sampled distractors, for 50 candidates. Title and description fields are truncated to 240 characters. MovieLens descriptions use literal native title/genre information.

The Beauty and Steam main protocol constructs the original 100-decision inputs and projects a fixed, source-selected subset to 20 decisions. This preserves the original attack construction before projection. MovieLens uses native 20-decision sequences. Each comparison uses 50 calibration sequences and two test sequences for each of five seed groups. The source projection is recorded in generated manifests. Its historical selection hash included source metadata, so exact historical cohorts require their original immutable artifacts; seeds alone do not identify those artifacts after relocation or refactoring.

Instruction injection edits the target item-text record and two seeded records in affected prompts. History manipulation inserts a target event and two recent-history filler events per affected user, retaining their identities on subsequent appearances. Rewriting modifies the target and two shared distractor descriptions using the Qwen2.5 adaptation of TextSimu. It uses top-10 attack feedback and evaluates the final recommendation at top-5. Attack generation and the auxiliary continuation model stay fixed when the recommendation backbone changes.

## Screening and selection

Screening evaluates promotion-sensitive rank changes caused by omissions, subtracts the historical influence reference, and penalizes the omitted fraction of context. Histories reset at sequence boundaries. `max_depth` bounds the number of jointly omitted identities; `beam_width` bounds the number of partial omission sets retained for expansion. With depth one, increasing beam width does not add deeper omission candidates.

Calibration uses the order statistic at `ceil((N + 1) * (1 - alpha))`, including the appended endpoint one. Final selection returns the screened ranking when its calibrated regime permits this. Otherwise, minimum-cost flow enforces an item-occurrence cap of `floor(eta * T)` and chooses exact-K outputs at minimum screened-rank cost. Items outside the screened top-K ranking receive rank K+1. Decisions with exactly K candidates have fixed outputs and are excluded from the cap. An infeasible sequence has no exact-K return.

The statistical guarantee assumes a fixed screening map and exchangeable calibration/test sequences. Chronological RQ1 splits and shifted RQ2 conditions are empirical comparisons; they do not establish this assumption automatically. RQ3 uses independently drawn calibration/test roles from the same fixed held-out generator. See [Statistical control](statistical-control.md).

## Aggregation

Recall@5 equals hit rate with one held-out positive per decision. NDCG@5 is zero on a miss and otherwise equals `1/log2(rank + 1)`. Both average over decisions within a sequence. Gmax is the largest item-wise fraction of decisions in which an item is returned but absent from its clean reference output; its denominator remains the full sequence length.

For each removal method, FPR is the number of omitted unmodified identities divided by all distinct unmodified identities in that sequence. Rates are averaged within seeds and then equally across seed groups. The paper-facing report does not use legacy pooled-count FPR or F1. Backbone and Spotlighting have no removal FPR.

Utility and promotion comparisons use a common all-method returned cohort with valid clean references. Return rates use all attempted sequences. Reports retain denominators and missing-output statuses. Optional RQ1 uncertainty resamples entire seed groups and is labeled descriptive; no significance stars are inferred from aggregate means.

## Timing

The main runner records stage times and model/cache statistics. Screening can require many diagnostic ranking calls; calibration and final selection alone are not the full cost of CREST. `benchmark_finalstage.py` measures fresh ranking generation from already prepared inputs plus CREST's final selection. It uses one fixed sequence per dataset, separate initially empty caches for each method and repetition, and CUDA synchronization. Model loading and the preparation stages are outside that specific timer. These measurements cannot establish full-pipeline real-time latency.
