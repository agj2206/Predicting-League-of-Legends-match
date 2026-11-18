# Decision log

Every non-trivial choice (filters, thresholds, K, minsup, features dropped) gets one entry
(CLAUDE.md §3.6). Newest at the bottom. **Append only** — never edit someone else's entry.
Never cite a number that did not come from executed code.

Format:

```
### D<nn> — <short title>
- **Date:** YYYY-MM-DD · **Author:** <member>
- **Decision:** what was chosen.
- **Alternatives:** what else was on the table.
- **Why:** the reason, with the evidence (notebook cell / number) behind it.
- **Affects:** which questions / files depend on this.
```

---

### D01 — Project pivoted from Olist review-risk to LoL early-game
- **Date:** 2026-11-18 · **Author:** A (Cesar)
- **Decision:** the group project is "how much of a ranked League match is decided at 10 minutes,
  and does that change with skill level", on two Kaggle 10-minute datasets (Diamond–Master and
  Challenger). The earlier Olist review-risk work was archived outside the repository.
- **Alternatives:** (a) continue with Olist (a complete data-quality audit already existed);
  (b) the "Name That LLM" open pool (212k passages) — rejected because TF-IDF and bag-of-words are
  not taught in this course, so the single most important transformation would have been
  off-syllabus, and the tabular data-quality work the rubric grades barely exists in it.
- **Why:** team decision, taken with the 20 Nov presentation date known.
- **Affects:** everything.

### D02 — Two tiers, not one
- **Date:** 2026-11-18 · **Author:** A (Cesar)
- **Decision:** use the Challenger 10-minute file alongside the Diamond–Master file.
- **Alternatives:** Diamond only (simpler, finishes faster).
- **Why:** the rubric rewards "collect dataset**s**" and "load **large** datasets", and the
  Diamond file alone is 9,879 rows. Two tiers also create the cross-tier question (Q4), which is
  what separates this from the many public notebooks on the same file.
- **Affects:** Q4, `load.py`, `clean.py`. Fallback if the merge proves slow: Diamond only for
  Friday, Challenger added back for the 6 December submission.

### D03 — Naive Bayes dropped from the model set
- **Date:** 2026-11-18 · **Author:** A (Cesar)
- **Decision:** Q1 compares Decision Tree, KNN and SVM. No Naive Bayes.
- **Alternatives:** include it as a fourth baseline.
- **Why:** measured, not assumed — "naive bayes"/"bayes" returns 1 hit across all ten lecture
  decks (a passing mention in L7). L4–L5 are decision trees, L6 is evaluation, L7 is KNN and SVM.
  CLAUDE.md §3.4 limits us to course techniques.
- **Affects:** Q1, `evaluate.py`, notebook 03.

### D04 — Challenger rows that are not 10-minute states are removed
- **Date:** 2026-11-18 · **Author:** A (Cesar, agent-assisted)
- **Decision:** drop Challenger rows where either team has `AvgLevel > 8.5`, `TotalGolds > 26,000`,
  or an inhibitor already destroyed. 2,498 rows flagged (9.46%); 23,911 kept. Combined with the
  9,879 Diamond games, the modelling table is 33,790 games.
- **Alternatives:** (a) keep every row — rejected, they are not 10-minute states and carry
  post-10-minute information; (b) drop the Challenger tier entirely — rejected, 97% of the problem
  is confined to one identifiable 9.5% slice; (c) filter on towers/inhibitors only — rejected, it
  caught 675 rows and left 52,754-gold rows behind.
- **Why:** executed evidence. `blueAvgLevel` jumps from 7.60 at p90 to 10.80 at p95 and
  `blueTotalGolds` from 21,101 to 33,073 over the same gap — a bimodal distribution, i.e. two
  populations in one file. The Diamond file, which is internally consistent, caps at level 8.0 and
  23,701 gold. A team cannot reach level 12.4 or 53,161 gold by minute ten, so the upper mode is
  late-game or full-game records mislabelled as 10-minute snapshots. After filtering, the kept rows
  cap at level 8.40 and 25,052 gold — consistent with Diamond, slightly higher as higher elo should be.
- **Affects:** `clean.py`, every question. Thresholds are deliberately loose (8.5 / 26,000) so the
  cut sits inside the empty gap between the two modes rather than trimming the legitimate tail.

### D05 — The filter is checked for target bias, not just plausibility
- **Date:** 2026-11-18 · **Author:** A (Cesar, agent-assisted)
- **Decision:** keep the D04 filter, having confirmed it does not move the target.
- **Alternatives:** weight or stratify to correct for selection.
- **Why:** `blueWins` is 0.4998 among kept rows vs 0.5011 across all Challenger rows — a shift of
  0.0013. Dropping rows on outcome-correlated variables is a real risk, so it was measured rather
  than assumed. No correction is needed.
- **Affects:** the report's limitations section must still state that the removed games were
  disproportionately decisive ones.

### D06 — `blueDragons` and `blueDragon` are the same concept (confirmed, not assumed)
- **Date:** 2026-11-18 · **Author:** A (Cesar, agent-assisted)
- **Decision:** map Challenger's `blueDragon` to Diamond's `blueDragons` in the canonical schema.
- **Alternatives:** treat them as different measures and exclude dragons from cross-tier features.
- **Why:** the raw files disagreed — Diamond takes values {0, 1}, Challenger {0, 1, 2} — which
  looked like a genuine semantic mismatch. It was not. All 51 Challenger rows showing 2 dragons sit
  inside the D04 contaminated slice; after that filter Challenger is {0, 1}, identical to Diamond.
  The mismatch was a symptom of the stage contamination, not a difference in meaning.
- **Affects:** `clean.py` column mapping, every cross-tier feature.

### D07 — `blueWardsDestroyed` maps to `blueWardKills`
- **Date:** 2026-11-18 · **Author:** A (Cesar, agent-assisted)
- **Decision:** treat them as the same concept.
- **Why:** distributions match closely — Diamond mean 2.82, p99 11, max 27; Challenger mean 3.89,
  p99 11, max 35. Same support, same shape, same order of magnitude.
- **Affects:** `clean.py` column mapping.

### D08 — Ward counts are capped, not dropped, and the cap is identical across tiers
- **Date:** 2026-11-18 · **Author:** A (Cesar, agent-assisted)
- **Decision:** winsorise `wards_placed` and `wards_destroyed` at a single physical bound applied
  to both tiers, rather than dropping the affected rows or dropping the columns.
- **Alternatives:** (a) drop the rows — rejected, they are otherwise ordinary games; (b) drop the
  ward columns — rejected, vision control is a real early-game lever and a legitimate feature;
  (c) a per-tier percentile cap — rejected, tier-dependent preprocessing would confound Q4.
- **Why:** after the D04 filter, Diamond's high-ward rows have entirely normal gold (16,533 vs
  16,398), level (7.0 vs 7.0) and CS (215 vs 218) — so they are valid games with one broken column,
  not broken games. Dropping them would discard good data over a single field. Both files converge
  on a tail reaching roughly 250 after filtering (Diamond 250, Challenger 257), which points to one
  shared upstream defect rather than random noise.
- **Affects:** `clean.py`. The report should note that ward features are the least trustworthy in
  the set, and the cap count must be reported.

### D09 — Features are signed differences, and `deaths_diff` is dropped by measurement
- **Date:** 2026-11-18 · **Author:** A (Cesar, agent-assisted)
- **Decision:** represent each contested quantity once, as `blue − red`, plus two genuine match
  totals (`total_gold_both`, `total_kills_both`). Build 15, drop `deaths_diff`, model on 14.
- **Alternatives:** (a) keep both sides of every concept — rejected, the mirrors are exact, so it
  doubles the column count without adding information and makes feature importances unreadable;
  (b) keep blue-side absolutes only — rejected, it discards the opponent's state, which is half of
  what decides a game.
- **Why:** measured in three stages by `features.audit_redundancy` and `features.independent_set`.
  The published Diamond file has 38 team columns containing 9 exact pairwise identities; the rank
  of the whole block is 25, lower than 38 − 9 because some dependencies span three columns
  (`blueGoldDiff = blueTotalGold − redTotalGold`). The difference collapse yields 15 features with
  one remaining exact dependency — `blue_deaths` equals `red_kills` in both tiers, so `deaths_diff`
  is identically `−kills_diff`. Dropping it leaves 14, with zero dependencies and full rank.
- **Affects:** Q1, Q2, Q3, Q4. Headline for the presentation: **38 published columns carry 14
  features' worth of information.**

### D10 — `tier` is a grouping column, never a feature
- **Date:** 2026-11-18 · **Author:** A (Cesar) — team decision
- **Decision:** `tier` stays in `games.parquet` for grouping and for Q4, and is excluded by
  `features.feature_columns()` from every model.
- **Alternatives:** include it so models can exploit skill-level differences.
- **Why:** a model allowed to see `tier` could learn which file a row came from rather than what
  the early game says, which would make Q4's transfer result meaningless.
- **Affects:** Q1 (B), Q4 (A). Agreed split: Q1 trains and tests within Diamond; Challenger is held
  out entirely as the transfer target; Q2 and Q3 run on the combined table.

### D11 — The two match totals are kept in the table, and expected to be dropped by B
- **Date:** 2026-11-18 · **Author:** A (Cesar, agent-assisted)
- **Decision:** `total_gold_both` and `total_kills_both` stay in `games.parquet` rather than
  being removed at the feature-building stage.
- **Alternatives:** drop them in `features.py` once they were found to carry no signal.
- **Why:** they measure how big the game was, not who is ahead, and the result does not care —
  correlation with the target 0.0077 and 0.0049, mutual information 0.0033 and 0.0000. That is a
  finding worth showing, and it is the cleanest statement of this project's central point: at ten
  minutes, only the gap predicts the winner. Removing them silently would hide the evidence.
  B's feature selection should discard them, and the fact that it does is a check that B's
  selection method works.
- **Affects:** Q1 (B) — expect 12 features after selection, not 14.

### D12 — Q4 holds Challenger out entirely, and splits Diamond so the comparison is honest
- **Date:** 2026-11-18 · **Author:** A (Cesar, agent-assisted)
- **Decision:** `transfer_report` fits on a 75% slice of Diamond and scores three ways — Diamond
  train, Diamond held-out, and the whole Challenger tier.
- **Alternatives:** fit on all of Diamond and compare in-sample Diamond against Challenger.
- **Why:** comparing an in-sample number against an out-of-sample one would confound the transfer
  gap with memorisation, and any drop could be attributed to either. The held-out Diamond row is
  the only fair baseline for the Challenger row.
- **Affects:** Q4, notebook 06. The report must also state that the two tiers are different
  populations of different sizes, so a higher score away is not evidence the model is "better at
  Challenger".
