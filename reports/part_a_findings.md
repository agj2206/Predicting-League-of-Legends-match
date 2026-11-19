# Part A: measured findings

Source: high_diamond_ranked_10min.csv; SHA-256 `82b2e23279097c2dac616e966eb2911711c428d41b6382b4cdb15da4892f9ca6`.
Observed source shape: 9,879 matches, 40 columns.
No observations removed or values corrected.

Missing values: 0; identical duplicate rows: 0; repeated gameId values: 0.
All numeric/binary/count/level and 13 arithmetic consistency checks passed.
Red wins: 4,949 (50.096%); Blue wins: 4,930 (49.904%).

Training: 7,903 × 12 predictors; labels (7903,). Red 3,959, Blue 3,944.
Test: 1,976 × 12 predictors; labels (1976,). Red 990, Blue 986.
80/20 stratified split, random_state=42, gameId ascending before splitting. No scaler, imputer, selector, or model fitted.

Retained original predictors: blueGoldDiff, blueExperienceDiff.
Engineered predictors: killsDiff, assistsDiff, minionsDiff, jungleMinionsDiff, wardsPlacedDiff, wardsDestroyedDiff, firstBloodDiff, dragonsDiff, heraldsDiff, towersDestroyedDiff. All are Blue minus Red.
Excluded original columns and individual reasons are listed in the notebook and feature_decisions().

Training-only Pearson associations (descriptive; not feature selection):
- blueGoldDiff: 0.511
- blueExperienceDiff: 0.489
- killsDiff: 0.479
- assistsDiff: 0.385
- minionsDiff: 0.307
- dragonsDiff: 0.229
- firstBloodDiff: 0.201
- jungleMinionsDiff: 0.178
- towersDestroyedDiff: 0.165
- heraldsDiff: 0.128
- wardsDestroyedDiff: 0.076
- wardsPlacedDiff: 0.016

Training-only predictor pairs with absolute Pearson correlation ≥ 0.75:
- blueGoldDiff / killsDiff: 0.918
- blueGoldDiff / blueExperienceDiff: 0.896
- killsDiff / assistsDiff: 0.831
- blueExperienceDiff / killsDiff: 0.824
- blueGoldDiff / assistsDiff: 0.758

Training-only extreme-value summaries (min to max; IQR flag counts are descriptive):
- blueTotalGold: 10730 to 23701; 87 flagged.
- redTotalGold: 11212 to 22732; 76 flagged.
- blueKills: 0 to 19; 68 flagged.
- redKills: 0 to 22; 53 flagged.
- blueTotalExperience: 10098 to 22224; 91 flagged.
- redTotalExperience: 10465 to 22269; 111 flagged.
- blueTotalMinionsKilled: 90 to 283; 54 flagged.
- redTotalMinionsKilled: 107 to 289; 72 flagged.
- blueTotalJungleMinionsKilled: 0 to 92; 134 flagged.
- redTotalJungleMinionsKilled: 4 to 92; 102 flagged.
- blueWardsPlaced: 5 to 185; 1310 flagged.
- redWardsPlaced: 6 to 276; 1341 flagged.

Ward counts reach 276 in training. Hundreds of wards at ten minutes are suspicious; the CSV cannot establish whether they are valid or extraction artifacts. All retained.
Gold, combat, experience and minion extremes may be legitimate; an IQR flag is not evidence of an error. No outlier filtering, clipping or replacement performed.
All eight figures use training data except the explicitly labeled full-source class-distribution figure.
Notebook assertions passed: source count, predictor exclusion, finite data, disjoint and complete partition, alignment, repeated split, feature transformation without the target, source unchanged and figure labels/files.
The automated test suite is separate; see README.md for its command. No model training, tuning, or evaluation performed.
