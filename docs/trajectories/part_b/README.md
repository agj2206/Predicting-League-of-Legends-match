# Part B — authentic Codex trajectories (privacy-redacted copies)

These files document the original AI-assisted planning, implementation, verification, and documentation of **Part B — Model Training** for CDS3004, *Predicting League of Legends Match Outcomes Using Early-Game Statistics*.

This README and `manifest.json` were created on **26 November 2026** to explain and audit the publication copies. They are new explanatory documentation, not original conversations. The `.redacted.jsonl` files derive directly from the original session records; their conversations have not been translated, improved, or reconstructed. Historical Italian and English text remains in its original language.

## Sessions and scope

Dates below use **Asia/Hong_Kong (UTC+08:00)**. Original JSONL timestamps remain in UTC and have not been changed. Each original filename maps to a copy with the same stem and the suffix `.redacted.jsonl`.

| Original session filename | Purpose | Original → published records |
|---|---|---:|
| `rollout-2026-11-18T18-49-11-01a1ee46-5e2c-7db1-bfd2-132d20edc74b.jsonl` | Earlier training-planning and course/dataset inspection; 18 November, 18:49–19:17 | 231 → 110 |
| `rollout-2026-11-19T00-58-24-01a1ef98-6415-7593-bd81-d9797d1f8c3b.jsonl` | **Primary Part B development**; 19 November, 00:58–19:00 | 358 → 165 |
| `rollout-2026-11-19T16-38-27-01a1f2f5-0a23-7932-b85b-83687bc1cd4c.jsonl` | **Supplementary automated approval review**, not a separate human development session; 19 November | 64 → 28 |

The first session was selected from its actual prompts: the operator identifies themselves as **Member 2**, responsible for Model Training, asks to inspect Lectures 4–7 and the Diamond dataset, and requests a training plan without implementation. Its initial workspace inspection and unsuccessful PDF-reading attempts are retained as part of that process. This is a planning precursor, not evidence that training was implemented on 18 November.

The primary session contains the Part B planning request, approval with **5 outer / 3 inner folds** and a `DummyClassifier` baseline, implementation of Decision Tree/KNN/SVM, scaling within pipelines, training-only nested CV, tuning, saved models/results, verification, documentation translation, and the final README correction about sharing the models. Original mistakes, failed attempts, tool outputs, and subsequent corrections remain where recorded.

The supplementary file identifies itself as `guardian_review`, has a `source.subagent.other` value of `guardian`, and names the primary thread as its parent. Its five review requests and five assistant decisions concern proposed actions in that development session. Messages with JSON role `user` in this file are **automatically generated review evidence**, not five new messages written by a student. The filename/session creation time is 16:38:27; the first persisted record and first review occur at approximately 16:54:51. These distinct timestamps are preserved.

## Attribution

The planning operator's Member 2 role is explicit in their original prompts. The other files are identified by their recorded project context and parent-session linkage. No real-name identity is inferred from a computer username, repository author, or planned speaking role. The automated reviewer is attributed to Codex's review process. These records do not establish work performed by Member 1 or Member 3, and an assistant's claim of completion is evidence of what it reported, not independent proof of that claim.

## Privacy transformations and omissions

The untouched originals remain separately in the local Codex session directory, outside this public repository. Their source-relative locations are recorded in `manifest.json` using `<CODEX_SESSION_ROOT>`; no personal absolute source path is published here.

Publication changes are limited to privacy redaction, removal of internal records, and JSON serialization:

- Personal project, workspace, course, dataset, home, Codex, and temporary paths use consistent placeholders such as `<PROJECT_ROOT>`, `<WORKSPACE_ROOT>`, `<COURSE_MATERIALS_ROOT>`, `<LOCAL_DATASETS_ROOT>`, `<USER_HOME>`, `<CODEX_HOME>`, and `<TEMP_ROOT>`. Plain paths, file URLs, URL-encoded paths, dictionary keys, command arguments, and nested JSON strings were inspected. Repository-relative paths such as `src/lol_model_training.py` are preserved.
- Local usernames in file ownership listings and text use `<LOCAL_USER>`. Creator user/account identifiers and private account metadata are removed.
- Three email occurrences in extracted course material use `<EMAIL_ADDRESS>`. One illustrative student identifier in course material uses `<STUDENT_ID>`; this is not an attribution to a project member.
- System/developer instructions, tool catalogs, runtime world/turn context, account quota/usage records, internal message metadata, and reasoning records (including encrypted reasoning) are omitted. Review-control instructions and permission-context fragments are replaced with explicit redaction markers; the proposed actions, recorded transcript evidence, and review outcomes remain.
- Two opaque binary PDF dumps from unsuccessful reading attempts are replaced with `[REDACTED_BINARY_PDF_DUMP]`. The associated calls, failure context, and later readable course explanations remain. These are not scientific results.
- Automatic environment messages and empty companion-page snapshots are omitted. No unrelated conversations or other projects are included. No substantive human prompt, assistant response, project implementation call, or associated recorded tool output within these selected sessions is omitted.

The inspection found no verified API keys, authentication tokens, passwords, or private keys. This describes the 26 November inspection, not a guarantee that every possible secret has been detected. Session, turn, message, and tool-call identifiers are retained for evidence linkage; they are not presented as account identifiers or access credentials.

The 27 November review masked remaining home paths, private project links and account references, and match-level samples and IDs in tool output. Aggregate results and recorded decisions remain. Record counts, timestamps, and tool-call linkage were verified against the earlier redacted copies. The manifest records the prior copy hash and follow-up edits under `release_review`. The earlier copies and manifests were preserved in a separate private local archive.

## Audit and validation

`manifest.json` records the original and current-copy SHA-256 values, record counts, original line number for every retained record, every omitted line with its reason, and the categories and JSON pointers of field changes. Original ordinals and timestamps are preserved; gaps are intentional. Raw timestamps, scientific values, implementation text, errors, and decisions have not been normalized or reordered. Nested JSON strings may be reserialized after their fields are redacted, so copies are not byte-identical raw exports and placeholder-containing commands are documentary evidence rather than runnable commands.

Validation checked every published JSONL line, monotonic timestamps, original record order, paired canonical tool calls/outputs, retention of substantive canonical human/assistant messages, exact structural comparison after the documented transformations, and unchanged original bytes. A separate recursive privacy scan inspected nested strings and keys. The final local check also compared existing project-file hashes and Git HEAD, branch, and index with the preparation baseline. No model training or evaluation was repeated for this preparation.

## Course submission requirement

The available official `L1_Intro.pdf`, page 14, requires **agent-use trajectories** and a short reflection, emphasizing what the agent did, what the human decided, and how results were verified. Page 12 refers to the syllabus rubric on Moodle. The later special tutorial and authenticated Moodle rubric were unavailable locally. The accessible material does **not specify unmodified raw JSONL**, so that requirement remains unverified. Team documentation referring to per-member trajectories is secondary evidence, not a substitute for the official rubric.

These copies are intended for public GitHub documentation. If the professor requires unmodified originals, provide the preserved originals through an authorized private submission channel instead of publishing sensitive raw logs. No reflection or missing interaction has been invented here.
