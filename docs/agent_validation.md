# Agent validation log

Every time the human or the agent catches an agent mistake — a wrong join, leakage, a bad metric,
a hallucinated column, an invented number — it gets an entry here (CLAUDE.md §3.7).
**This log is graded.** An empty one reads as "we never checked". Honesty is the point: do not
quietly fix and move on. **Append only** — never edit someone else's entry.

Format:

```
### V<nn> — <short title>
- **Date:** YYYY-MM-DD · **Caught by:** human / agent / test
- **What happened:** the mistake, concretely.
- **How it was detected:** the check, assert, or review that surfaced it.
- **Fix:** what changed, and where.
- **Guardrail:** what now prevents a repeat (an assert, a rule, a review step).
```

---

### V01 — Agent asserted a dataset's schema from its description instead of the file
- **Date:** 2026-11-18 · **Caught by:** agent (before any code was written)
- **What happened:** while scoping the project, the agent stated the Diamond file's column list
  and row count from the Kaggle page and web search results, not from the CSV. The data had not
  been downloaded at that point, so none of it was verified.
- **How it was detected:** checking `data/raw/` and finding no LoL file there at all.
- **Fix:** every shape, column and dtype claim is now deferred to notebook 01, which measures it.
  CLAUDE.md §2 states the rule explicitly.
- **Guardrail:** a dataset description is not evidence. The two tiers were built by different
  authors, so their schemas must be compared by reading both files, never by assuming they match.

### V02 — A published dataset's own labelling was taken at face value
- **Date:** 2026-11-18 · **Caught by:** agent, while profiling before writing any loader
- **What happened:** the plan treated `Challenger_Ranked_Games_10minute.csv` as a 10-minute
  snapshot because that is what the filename, the Kaggle page and our own CLAUDE.md said. It is not.
  About 9.5% of its rows record a much later game state — up to level 12.4 and 53,161 team gold,
  with inhibitors destroyed. An inhibitor cannot fall by minute ten.
- **How it was detected:** profiling the columns the merge would need, and noticing that
  `blueInhibitor` was non-zero in 0.62% of rows. That prompted checking the physical ceilings, which
  showed `blueAvgLevel` jumping from 7.60 at p90 to 10.80 at p95 — a gap with no mass in it, so two
  populations, not a heavy tail.
- **Why it mattered:** those rows carry post-10-minute information (towers, inhibitors, doubled
  gold) that is enormously predictive of the winner. Training on them would have inflated Q1's
  scores with information unavailable at the decision point — leakage introduced by the data
  source rather than by us.
- **Fix:** the D04 filter in `clean.py`, with the Diamond file used as a known-clean reference for
  what a 10-minute state looks like.
- **Guardrail:** a filename is not a measurement. Before merging any two sources, check that the
  same column means the same thing in both, against a physical bound where one exists. The first
  version of the filter (towers and inhibitors only) caught 675 rows and left 52,754-gold rows in —
  so the guardrail is to re-check the extremes *after* filtering, not just before.

### V03 — Two anomalies were nearly treated as two problems
- **Date:** 2026-11-18 · **Caught by:** agent, by testing whether one fix resolved the other
- **What happened:** the plan listed the ward outliers and the dragon range mismatch as separate
  data-quality items needing separate handling, alongside the D04 stage contamination. Treating
  them separately would have meant three ad-hoc rules.
- **How it was detected:** checking the extreme-ward rows' other columns, which showed median gold
  31,071 against 16,604 — the signature of the D04 contaminated slice, not an independent defect.
  Re-running the ward and dragon statistics *after* the D04 filter then confirmed it: Challenger's
  ward maximum fell from 1,318 to 257 and its dragon range collapsed from {0,1,2} to {0,1}.
- **Fix:** one root cause, documented once (D04). The dragon mismatch needs no rule at all (D06).
  Only the residual ward tail, which survives in both files and belongs to otherwise-normal games,
  needs its own treatment (D08).
- **Guardrail:** before writing a rule for an anomaly, check whether a rule already written explains
  it. Re-measure after each filter rather than stacking filters on stale statistics.

### V04 — A carried-forward module kept an import from the abandoned project
- **Date:** 2026-11-18 · **Caught by:** agent, reading the file before using it
- **What happened:** `viz.py` was copied from the archived Olist project because its figure style
  and 150-dpi save rule are project-agnostic. Its `figures_dir()` still contained
  `from olist.load import repo_root`. The module imports fine — the bad import is inside a
  function — so it would have raised `ModuleNotFoundError` only at the first `save_fig()` call,
  i.e. after a notebook had already run its analysis.
- **How it was detected:** reading the helper signatures before writing figure code, rather than
  calling them and waiting for the traceback.
- **Fix:** changed to the relative `from .load import repo_root`.
- **Guardrail:** a file carried across projects is not verified by the fact that it imports. Read
  what it references, and exercise the function, not just the module.

### V05 — An interpretation helper assumed transfer could only get worse
- **Date:** 2026-11-18 · **Caught by:** agent, reading its own output against the numbers
- **What happened:** `transfer.diagnose()` branched on the *absolute* size of the metric change
  and then described it in words written for degradation. On the placeholder model, where
  transfer improved (ROC-AUC +0.018, accuracy +0.024), it reported "the threshold needs re-fitting
  per tier" — advice that makes no sense when the away tier scores higher.
- **How it was detected:** reading the generated sentence against the table above it, rather than
  accepting it because the function ran without error.
- **Fix:** the branches are sign-aware, and an improvement is now described as such, with the
  caveat that two different populations are not a like-for-like contest.
- **Guardrail:** a helper that writes prose is as capable of being wrong as one that computes a
  number, and it fails silently because the text always reads plausibly. Check generated wording
  against the data it describes.

### V06 — The agent predicted a result instead of measuring it
- **Date:** 2026-11-18 · **Caught by:** agent, on running the check it had already written down
- **What happened:** D11 asserted that every reasonable feature-selection method would discard
  `total_gold_both` and `total_kills_both`, and proposed using that as a check on B's method. The
  claim was extrapolated from one statistic — near-zero correlation with the target — and written
  into the decision log as though it were established.
- **How it was detected:** actually running the three selection methods. They agree on dropping
  nothing, and tree importance ranks `total_gold_both` 4th of 14.
- **Why it mattered:** the false claim was written into a handoff document. B would have read
  "expect 12 features after selection", found 14, and had to work out which of the two sources to
  believe — or worse, forced the expected answer.
- **Fix:** D13 supersedes the prediction and records what the measurement showed, including the
  interaction that explains it and the nested-CV result showing it is not load-bearing.
- **Guardrail:** this repeats V02's lesson in a new costume. A decision log entry states what was
  measured and what was chosen; a prediction about what a later step will find does not belong in
  it unless it is labelled as untested.
