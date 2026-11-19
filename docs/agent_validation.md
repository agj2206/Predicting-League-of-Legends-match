# Agent validation log

Every time the human or the agent catches an agent mistake — a wrong join, leakage, a bad metric,
a hallucinated column, an invented number — it gets an entry here (CLAUDE.md §3.7).
**This log is graded.** An empty one reads as "we never checked". Honesty is the point: do not
quietly fix and move on. **Append only.**

Format:

```
### V<nn> — <short title>
- **Date:** YYYY-MM-DD · **Caught by:** human / agent / test
- **What happened:** the mistake, concretely.
- **How it was detected:** the check, assert, or review that surfaced it.
- **Fix:** what changed, and where.
- **Guardrail:** what now prevents a repeat.
```

---

### V01 — A dataset's schema was stated from its description instead of the file
- **Date:** 2026-11-18 · **Caught by:** agent, before any code was written
- **What happened:** while scoping the project, the agent stated the dataset's column list and row
  count from the Kaggle page and from web search results. The data had not been downloaded, so
  none of it was verified.
- **How it was detected:** checking `data/raw/` and finding no file there at all.
- **Fix:** every shape, column and dtype claim was deferred until the file could be measured.
  CLAUDE.md §2 now states the rule, and `consistency_checks()` enforces the contract in code.
- **Guardrail:** a dataset description is not evidence. The schema is asserted in
  `tests/test_preprocessing.py`, so a changed source fails loudly instead of silently.

### V02 — A file was trusted because of its name
- **Date:** 2026-11-18 · **Caught by:** agent, while profiling a second candidate dataset
- **What happened:** a Challenger-tier file named `..._10minute.csv` was treated as a 10-minute
  snapshot because its filename, its Kaggle page and our own planning document all said so. It is
  not: roughly 9.5% of its rows record a much later game state, up to average level 12.4 and
  53,161 team gold, with inhibitors destroyed. An inhibitor cannot fall by minute ten.
- **How it was detected:** noticing that an `Inhibitor` column was non-zero in 0.62% of rows, then
  checking quantities with a hard physical ceiling. Average level jumped from 7.60 at p90 to 10.80
  at p95 with no mass in between — two populations in one file, not a heavy tail.
- **Why it mattered:** those rows carry post-10-minute information that is enormously predictive
  of the winner. Training on them would have inflated Q1's scores with information unavailable at
  the decision point — leakage shipped by the data source rather than introduced by us.
- **Fix:** that tier was ultimately dropped for scope (D02), but the finding is why the Diamond
  file was checked the same way rather than trusted. It passes: 13 arithmetic consistency checks,
  no missing values, no duplicate ids, every quantity inside a plausible 10-minute range.
- **Guardrail:** a filename is not a measurement. Before using any source, check that its columns
  mean what they claim, against a physical bound where one exists.

### V03 — A module carried an import from an abandoned project
- **Date:** 2026-11-18 · **Caught by:** agent, reading the file before using it
- **What happened:** a plotting helper was reused from earlier work. Its `figures_dir()` still
  imported from the old project's package. The module imported fine because the bad import sat
  inside a function, so it would have raised `ModuleNotFoundError` only at the first save call —
  after a notebook had already run its analysis.
- **How it was detected:** reading the helper's source before calling it, rather than waiting for
  the traceback.
- **Fix:** corrected to a relative import at the time; that helper is no longer in the repository.
- **Guardrail:** a file carried across projects is not verified by the fact that it imports. Read
  what it references and exercise the function, not just the module.

### V04 — A helper that writes prose assumed its result could only go one way
- **Date:** 2026-11-18 · **Caught by:** agent, reading its own output against the numbers
- **What happened:** a function that interpreted a model-comparison table branched on the absolute
  size of each metric change, then described it in words written for the case where performance
  degrades. When performance improved, it confidently recommended an action that made no sense for
  an improvement.
- **How it was detected:** reading the generated sentence against the table above it, instead of
  accepting it because the function ran without error.
- **Fix:** the branches were made sign-aware before the helper was used anywhere.
- **Guardrail:** a helper that writes prose is as capable of being wrong as one that computes a
  number, and it fails silently because the text always reads plausibly. Check generated wording
  against the data it describes.

### V05 — The agent predicted a measurement instead of taking it
- **Date:** 2026-11-18 · **Caught by:** agent, on running the check it had already written down
- **What happened:** a decision-log entry asserted that every reasonable feature-selection method
  would discard two particular features, and proposed using that as a check on a teammate's work.
  The claim was extrapolated from a single statistic — near-zero correlation with the target — and
  written into a handoff document as though established.
- **How it was detected:** actually running the three selection methods. They agreed on discarding
  nothing, and tree importance ranked one of the two features 4th of 14, because it interacts with
  the others rather than predicting on its own.
- **Why it mattered:** the false claim was in a document a teammate was meant to work from.
- **Fix:** superseded by a corrected entry recording what the measurement showed.
- **Guardrail:** a decision log states what was measured and what was chosen. A prediction about
  what a later step will find does not belong in it unless it is labelled as untested.

### V06 — Descriptive statistics were computed on the full dataset, not on training only
- **Date:** 2026-11-19 · **Caught by:** human review, comparing two independent Part A builds
- **What happened:** one build of the preprocessing stage computed feature-target correlations and
  the redundancy audit over the entire dataset, then split afterwards. Feature descriptions were
  therefore informed by rows that later served as the held-out test set.
- **How it was detected:** reviewing a second, independently written Part A which split first and
  reported every statistic as training-only. The two agreed on the numbers but not on the method,
  and the stricter method is the correct one.
- **Why it mattered:** it is a mild form of looking at the test set. Decisions made after seeing
  test-set statistics are no longer independent of it, and the final held-out score stops being an
  honest estimate.
- **Fix:** the retained Part A splits before computing anything, and every descriptive statistic
  is training-only, with the one full-source figure explicitly labelled (D04).
- **Guardrail:** the split comes first, always — before correlations, before audits, before
  figures. "Just describing the data" is not an exemption.
