# Teo Dataset

**Version:** 0.2
**Last updated:** 2026-06-30

| Category    | Count | Languages     |
|-------------|-------|---------------|
| coding      | 30    | en, uz, mixed |
| debugging   | 25    | en, uz, mixed |
| deployment  | 25    | en, uz, mixed |
| greetings   | 20    | en, uz, mixed |
| teaching    | 21    | en, uz, mixed |
| security    | 20    | en, uz, mixed |
| personality | 15    | en, uz, mixed |
| refusals    | 15    | en, mixed     |
| humor       | 10    | en, mixed     |
| **Total**   | **181** |             |

## Next Goal: 5,000+ samples

Empty categories still needed (next priority, roughly in this order):
`conversation` (30), `linux` (40), `python` (40), `git` (30), `fastapi` (30),
`docker` (30), `project` (30), `random` (20).

## Rules

- Schema is fixed — see `validate_dataset.py`, run it after every batch.
- IDs are globally unique across every file in `train/`. Never reuse one.
- Write messy, not textbook. Typos, short messages, frustration, incomplete
  questions, mixed en/uz in the same message — that's what real users send.
- `source: "human"` for anything written by a person (you, classmates,
  roommates). Don't fake this field.
- Never delete a sample. If it's wrong or deprecated, move it to `archive/`.

## Curriculum

| Phase | Data used               | Goal                              |
|-------|--------------------------|------------------------------------|
| A     | `difficulty: "easy"`     | basic communication                |
| B     | + `"medium"`             | common dev tasks                   |
| C     | + `"hard"`               | complex problems                   |
| D     | + `"expert"`             | architecture / research-level Q&A  |
