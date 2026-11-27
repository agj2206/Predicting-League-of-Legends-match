# Predicting League of Legends Match Outcomes Using Early-Game Statistics

CDS3004 Data Mining Group Project

**Research question.** Can we predict the winning team in a League of Legends match using only statistics available at the 10-minute mark?

League of Legends is a competitive game in which two teams, Blue and Red, try to destroy the opposing team's Nexus. Gold, experience, combat and map objectives describe the early state of the match. This project tests how much predictive information that state contains about the eventual winner.

## Dataset

The source is `high_diamond_ranked_10min.csv` from [League of Legends Diamond Ranked Games (10 min), published by bobbyscience on Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min). The source describes ranked solo-queue games from high Diamond to low Master. The exact CSV used here has **9,879 matches and 40 columns**. The binary outcome `blueWins` is 1 for a Blue win and 0 for a Red win. Blue won 4,930 matches and Red won 4,949.

| Partition | Matches | Red wins | Blue wins |
| --- | ---: | ---: | ---: |
| Training | 7,903 | 3,959 | 3,944 |
| Held-out test | 1,976 | 990 | 986 |

## Project workflow

### Part A — Data preprocessing, feature engineering and EDA

Part A verifies the source checksum, schema, missing values, duplicate identifiers and Blue–Red arithmetic. All records were retained. Each match is represented by **12 Blue-minus-Red predictors** covering gold, experience, combat, farming, vision and objectives. The feature policy was fixed before examining feature–outcome associations. `gameId` orders rows before the stratified 80/20 split (seed 42) and serves as a row index. Predictor-level exploratory analysis uses the training partition. Gold difference had the highest training Pearson correlation with the outcome (0.511), followed by experience difference (0.489) and kills difference (0.479).

[Part A notebook](notebooks/01_data_preprocessing.ipynb) · [Preprocessing and feature policy](README_PART_A.md) · [Findings](reports/part_a_findings.md) · [Target associations](reports/figures/07_training_target_associations.png) · [Predictor correlations](reports/figures/08_training_predictor_correlation.png)

### Part B — Model training and nested cross-validation

Part B compares a majority-class Dummy baseline, Decision Tree, k-nearest neighbors (KNN) and support vector machine (SVM). On the training set, stratified nested cross-validation uses **5 outer folds and 3 inner folds** with seed 42. The inner folds tune hyperparameters, and the outer folds estimate validation performance. KNN and SVM learn standardization inside their pipelines on each fit. The family with the highest mean outer-fold accuracy is selected before the final three-fold searches and training-set refits. The chosen pipeline is a **linear SVM with `C=10`**.

[Part B notebook](notebooks/02_model_training.ipynb) · [Training protocol](README_PART_B.md) · [Findings](reports/part_b_findings.md) · [Saved CV results](reports/model_training/diamond_5x3_seed42/)

### Part C — Model evaluation and interpretation

Part C scored all four frozen pipelines on the original held-out split. The preselected SVM remained the primary model. Evaluation reports accuracy, Blue-class precision, recall and F1, ROC-AUC and confusion matrices. The SVM's ranking scores come from its decision function. Gold difference, experience difference and dragon difference have the largest absolute standardized SVM coefficients. These coefficients describe the fitted decision score conditional on the other features.

[Part C notebook](notebooks/03_model_evaluation.ipynb) · [Evaluation protocol](README_PART_C.md) · [Findings and figures](reports/part_c_findings.md) · [Saved evaluation results](reports/model_evaluation/diamond_5x3_seed42/)

## Main results

| Model | Mean outer CV accuracy | Held-out accuracy | Held-out ROC-AUC |
| --- | ---: | ---: | ---: |
| Dummy (majority class) | 50.09% | 50.10% | 0.5000 |
| Decision Tree | 71.58% | 72.77% | 0.7994 |
| KNN | 71.62% | 72.62% | 0.8081 |
| **Linear SVM (`C=10`)** | **72.95%** | **73.58%** | **0.8149** |

The SVM made **1,454 correct** and **522 incorrect** predictions. Its held-out accuracy exceeded the majority baseline by **23.48 percentage points**. Ten-minute statistics supported above-baseline prediction for the evaluated sample.

[CV and test accuracy](reports/figures/part_c/diamond_5x3_seed42/cv_test_accuracy.png) · [Confusion matrices](reports/figures/part_c/diamond_5x3_seed42/confusion_matrices.png) · [ROC curves](reports/figures/part_c/diamond_5x3_seed42/roc_curves.png) · [SVM coefficients](reports/figures/part_c/diamond_5x3_seed42/svm_standardized_coefficients.png)

## Installation

Use **Python 3.11**. From the repository root on macOS or Linux, run the following commands. On Windows, use `py -3.11` to create the environment and `.venv\Scripts\python.exe` for subsequent commands.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-part-b.txt
.venv/bin/python -m pip check
```

[`requirements.txt`](requirements.txt) pins Part A's direct dependencies. [`requirements-part-b.txt`](requirements-part-b.txt) includes those pins and adds the versions needed to load the saved models. The combined installation covers Part C as well. Matplotlib supplies Pillow for the figure auditor, and the notebook stack supplies `jupyter_client` and IPython. The recorded training environment used Python 3.11.5. [Environment record](reports/model_training/diamond_5x3_seed42/environment_versions.txt)

## Reproduction and verification

- **Inspect the saved experiment.** The executed notebooks, aggregate JSON and figures are available in the repository. The Part B and Part C notebooks can be regenerated from their saved inputs without the dataset, model fitting or held-out inference. Their scripts update the notebook files in place. Use a disposable copy to preserve the saved versions.

  ```bash
  .venv/bin/python -B scripts/execute_training_notebook.py
  .venv/bin/python -B scripts/execute_evaluation_notebook.py
  ```

- **Reproduce Part A.** Download the named CSV from the [Kaggle data card](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min). Extract it locally to `data/raw/high_diamond_ranked_10min.csv`, or point `LOL_DATA_PATH` to an absolute file path with that filename. The loader also accepts a unique match in a sibling `Local_Datasets` directory. The expected SHA-256 is `82b2e23279097c2dac616e966eb2911711c428d41b6382b4cdb15da4892f9ca6`. Keep the original CSV, split membership and per-match predictions local. Run these commands from the repository root. The notebook script refreshes the Part A notebook, EDA figures and findings in place, so use a disposable copy when checking the published displays.

  ```bash
  .venv/bin/python -B -m unittest discover -s tests -p test_preprocessing.py -v
  .venv/bin/python -B scripts/execute_notebook.py
  ```

- **Training and evaluation provenance.** The [Part B guide](README_PART_B.md) documents the original training recipe and the use of a separate run ID for an independent training reproduction. The completed Part C evaluation is documented in the [Part C guide](README_PART_C.md). Its display-only mode reads saved aggregate results. The historical preflight and prediction-cache audit have additional local prerequisites and file-hash checks, detailed there. The test suite includes Part B tests that fit models, so use targeted tests when verifying frozen artifacts.

## Repository structure

| Path | Contents |
| --- | --- |
| [`notebooks/`](notebooks/) | Three executed notebooks for Parts A, B and C |
| [`src/`](src/) | Preprocessing and validation, nested CV and training, evaluation and interpretation |
| [`scripts/`](scripts/) | Notebook execution, training, evaluation, result rendering and verification entry points |
| [`tests/`](tests/) | Tests for all three stages |
| [`models/part_b/diamond_5x3_seed42/`](models/part_b/diamond_5x3_seed42/) | Four fitted pipelines from the recorded training run |
| [`reports/`](reports/) | Findings, figures, nested CV search results and saved evaluation results |
| [`docs/`](docs/) | Decisions, agent validation and the trajectory index |
| [`data/`](data/) | Placeholders for local raw and processed data |

## Limitations

- The sample covers a narrow ranked population. The CSV contains no player identifiers, match dates, patch or region fields. Repeated-player independence and performance on other ranks, regions or future patches remain unverified.
- Suspiciously large ward counts were retained. The dataset provides no match timelines to resolve them. Relative predictors discard some information about absolute game tempo and tied objective states.
- The results describe observational prediction. Correlations and coefficients among correlated predictors do not establish causal effects.
- The experiment uses one holdout and finite hyperparameter grids. The selected model's CV mean can be optimistic, and fold variability does not establish statistical superiority. SVM decision scores are uncalibrated.

## Group contributions and agent records

- **Part A.** The work plan assigns preprocessing, validation, splitting and EDA to Cesar. The repository records this implementation in commit `273860b` under the Git author name `leyroda`. The Claude Code session archive is attributed to Cesar. [Part A sessions](docs/trajectories/README.md#part-a--cesar)
- **Part B.** The work plan assigns model training and tuning to Andrea. Commit `98d791d` records the implementation under the Git author name Andrea. The Codex planning operator identifies as Member 2 in the session. [Part B sessions](docs/trajectories/part_b/README.md)
- **Part C.** The work plan assigns evaluation to Christian. The implementation and completion commits (`9203ee8` and `1b40e46`) list Andrea as Git author. The completion commit also credits Natalia as a co-author. The Codex records document the local operator's evaluation interactions and automated review. The records do not independently establish the operator's identity or Christian's implementation contributions. [Part C sessions](docs/trajectories/part_c/README.md)

Individual presentation responsibilities are unverified in the available project records. The [work plan](SPRINTS.md) records planned assignments, and the [trajectory index](docs/trajectories/README.md) explains the scope and attribution of the archived sessions. The [decision log](docs/decisions.md) and [agent validation notes](docs/agent_validation.md) record choices and corrections during development.

## Publication status

The working tree contains privacy-reviewed trajectory copies. Earlier versions remain reachable in published Git history. Kaggle identifies the exact source dataset as CC0. The fitted KNN and SVM retain recoverable training observations, so their inclusion in a public release requires a distribution decision. See the [publication review](docs/publication_review.md) and the [trajectory index](docs/trajectories/README.md).
