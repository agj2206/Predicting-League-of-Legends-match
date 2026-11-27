# Part C — authentic Codex trajectories (privacy-redacted excerpts)

These files document the original AI-assisted **Part C — Model Evaluation and Interpretation** for CDS3004, *Predicting League of Legends Match Outcomes Using Early-Game Statistics*.

This README and `manifest.json` were created on **26 November 2026** as explanatory publication documentation. They are not original dialogue. The `.redacted.jsonl` files are privacy-redacted, explicitly scoped excerpts of original session logs. Conversations retain their original Italian or English wording, including recorded errors and corrections; no missing interaction has been reconstructed.

## Sessions and scope

Dates below use **Asia/Hong_Kong (UTC+08:00)**. Original JSONL timestamps remain unchanged in UTC. Each original filename maps to a copy with the same stem and the suffix `.redacted.jsonl`.

| Original session filename | Purpose | Original → published records |
|---|---|---:|
| `rollout-2026-11-19T22-00-42-01a1f41c-11c2-78b2-8f05-e07d430e656c.jsonl` | **Primary evaluation development**, first three turns; 19 November, 22:00–22:45 | 494 → 157 |
| `rollout-2026-11-19T22-00-42-01a1f41c-11de-79f2-b40f-af968eb2da67.jsonl` | **Supplementary automated approval review**, first eight reviews; 19 November, approximately 22:01–22:43 | 129 → 44 |

Selection was based on content and provenance, not just dates. The primary session explicitly requests Part C, names `src/lol_model_evaluation.py`, the frozen Part B models and the original 1,976-match test partition, and records these stages:

1. Compatibility inspection of Parts A/B and the proposed evaluation plan, with no test inference authorized yet.
2. Approval to implement the evaluation workflow, synthetic tests, and numerical SVM coefficient checks while the real test evaluation remains withheld.
3. Explicit authorization of final evaluation, the recorded held-out workflow, confusion matrices, ROC curves, comparison with nested CV, interpretation, notebook/report work, and verification.

The evidence includes the original `SVC.coef_` numerical warnings, coefficient reconstruction, preliminary tests, plotting/kernel problems and corrections, process-management attempts, the authorized final evaluation, and checks of saved results. Scientific metrics, code, model parameters, and methodological explanations are preserved. Four unique embedded scientific PNG images were retained and their metadata checked; the metadata identified Matplotlib software and did not expose a personal path.

The automated reviewer is identified by `guardian_review`, `source.subagent.other=guardian`, and its parent-thread link to the primary session. Its first eight requests and outcomes concern Part C actions, including the historical authorized fetch, notebook/kernel handling, and notebook execution after evaluation. JSON role `user` in this file denotes **automatically assembled review evidence**, not a human student's new prompt. The filename/session metadata creation time is 22:00:42; its first persisted review record occurs at approximately 22:01:08.

### Explicit scope boundary

The original primary file has a fourth turn starting around **23:02** for a complete project documentation/presentation handoff package. That later task is outside the evaluation-development scope:

- Primary source **lines 1–345** are eligible for inclusion, subject to privacy/internal-record filtering. **Lines 346–494 (149 records)** are omitted as the later packaging turn, including its preceding settings event.
- Supplementary source **lines 1–96** are eligible. **Lines 97–129 (33 records)** are omitted as the corresponding later packaging-review boundary/settings and reviews.

The separate presentation-design sessions are not Part C development trajectories and are excluded. The excerpts are therefore not described as complete copies of their original multi-purpose sessions. All three substantive Part C human prompts, fifteen canonical assistant messages, and all 51 canonical tool calls with their 51 outputs in the in-scope primary portion remain. Eight automated review requests and eight review responses remain in the supplement.

## Attribution

These files record the interaction of the local operator with Codex and the associated automated reviewer. The earlier Part B planning evidence contains the operator's self-identification as **Member 2, responsible for Model Training**. Part C's actual prompts and actions show that the evaluation interaction occurred in this local project context; they do not independently verify the operator's real-name identity. The planned allocation of evaluation to Member 3 does **not** establish that Member 3 conducted these sessions. No Part C interaction is reassigned to Member 3 or a named teammate. The reviewer performed automated assessments, not an independent student contribution.

## Privacy transformations and omissions

The originals remain unchanged in the local Codex session directory outside this public repository. `manifest.json` gives their filenames and locations relative to `<CODEX_SESSION_ROOT>`.

- Personal absolute project/workspace/course/dataset/home/Codex paths use the same placeholders as Part B: `<PROJECT_ROOT>`, `<WORKSPACE_ROOT>`, `<COURSE_MATERIALS_ROOT>`, `<LOCAL_DATASETS_ROOT>`, `<USER_HOME>`, `<CODEX_HOME>`, and `<TEMP_ROOT>`. Redaction covers command arguments, outputs, file URLs, encoded paths, nested JSON strings, and dictionary keys. Useful repository-relative paths such as `src/lol_model_evaluation.py` remain intact.
- Local usernames become `<LOCAL_USER>`. Creator user/account identifiers, internal message metadata, account quota/usage, runtime world/turn context, system/developer instructions, tool definitions, and reasoning records including encrypted reasoning are removed.
- Automated review-control instructions and internal permission-context fragments receive explicit redaction markers. Review evidence, proposed actions, decisions, rationale, and relevant errors remain.
- Automatic environment and empty companion-page messages are omitted. The later packaging ranges described above are omitted for scope. No unrelated private conversation is included.
- No verified API key, authentication token, password, personal email, student ID, local machine name, or private-key material was found in the included Part C evidence. Their detection patterns were checked; this does not guarantee detection of every possible secret. Operational session, turn, message, and call IDs remain for evidence linkage and are not account credentials.

No numeric result, hyperparameter, substantive Part C user decision, or relevant recorded implementation/testing interaction was removed for presentation quality. Original truncation notices remain: text absent from the original log is not reconstructed. Serialization may change JSON whitespace/escaping, and commands with placeholders are documentary evidence rather than runnable commands.

The 27 November review masked remaining home paths, private project links, and account or application references in tool and review output. All four scientific PNGs remain byte-identical. Record counts, timestamps, and tool-call linkage were verified against the earlier redacted copies. The manifest records the prior copy hash and follow-up edits under `release_review`. The earlier copies and manifests were preserved in a separate private local archive.

## Audit and validation

`manifest.json` contains original/copy SHA-256 values, full-original and excerpt record counts, original line mappings, per-record omission reasons, field-change categories/JSON pointers, and timestamps. Original ordinals and timestamp order are preserved without sorting. Its newly created audit entries are explanatory data, not inserted conversation records.

Every published JSONL line was parsed. Checks covered unchanged timestamps, chronological order, complete canonical in-scope messages and paired calls/outputs, structural comparison with the original after the documented transformations, nested privacy scanning, and unchanged original logs. Final checks also compare existing project-file hashes and Git HEAD, branch, and index against the baseline. No model training, prediction, final evaluation, Git commit, push, branch switch, or merge was performed during this preparation.

## Course submission requirement

The available official `L1_Intro.pdf`, page 14, requires agent-use trajectories and a short human–agent collaboration reflection. Page 12 refers to the syllabus rubric on Moodle. The later special tutorial and authenticated Moodle rubric were unavailable locally; **a requirement for unmodified raw logs could not be verified**. The accessible introduction does not specify raw JSONL or require public publication of sensitive originals.

If unmodified originals are required, supply the preserved full originals through an authorized private submission channel. These public excerpts and their audit manifest do not replace a stricter private-submission requirement, establish another member's contribution, or supply a fabricated reflection.
