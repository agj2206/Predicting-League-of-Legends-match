# Agent trajectories

Each member's agent-use history, as required by the course (Special Tutorial, slide 15).
Together with the commit history this is how individual workload is assessed (slide 9).

| Member | Files | Tool |
|---|---|---|
| Cesar | `cesar/` | Claude Code |
| Andrea | — | Codex |
| Christian | — | — |

## Cesar

- `2026-11-14_initial_exploration.jsonl` — the first session, on an e-commerce review-risk
  dataset that was explored to a completed data-quality audit and then dropped. Kept because
  decision D01 cites it as a considered alternative, and the record shows that work was real
  rather than hypothetical.
- `2026-11-18_to_11-26_lol_project.jsonl` — the League of Legends project: dataset selection,
  Part A preprocessing, the data-quality findings, and the project documentation.

These are raw JSONL session logs: one JSON object per line, holding the full conversation
including every tool call and its result. They are not summaries and have not been edited.

## Reading them

```
python -c "
import json
for line in open('cesar/2026-11-18_to_11-26_lol_project.jsonl'):
    r = json.loads(line)
    print(r.get('type'), str(r.get('message', ''))[:160])
"
```

Mistakes caught during these sessions are written up in `../agent_validation.md`, which is the
readable version of what these logs contain.
