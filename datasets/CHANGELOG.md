# Dataset update — English-only + expansion

Built against `Teo_v4_withdata` (the newer local copy). Only `datasets/train/` and
`datasets/tokenizer/vocab.json` were touched — no code changed.

## What happened

1. Removed every sample where `language` was `uz` or `mixed` (39 total, out of 218).
   No `ru` samples existed. All removed rows are gone, not archived — pull them back
   from git history if you ever want them.
2. Added 244 new English samples, `source: "generated"` (not `"human"` — since I wrote
   these, not you or a classmate, per your own README rule not to fake that field).
   IDs continue from 243 (previous max was 242), all unique.
3. Added two new files, `docker.jsonl` and `conversation.jsonl` — both categories were
   already recognized by `quick_add.py` but had zero samples yet.

## Before → after

|           file           | before | removed(non-en) | kept | added | final |
|--------------------------|--------|-----------------|------|-------|-------|
| coding.jsonl             |   34   |        7        |  27  |   25  |   52  |
| debugging.jsonl          |   27   |        6        |  21  |   20  |   41  |
| deployment.jsonl         |   25   |        5        |  20  |   18  |   38  |
| personality.jsonl        |   25   |        4        |  21  |   15  |   36  |
| teaching.jsonl           |   21   |        4        |  17  |   15  |   32  |
| security.jsonl           |   20   |        4        |  16  |   15  |   31  |
| greetings.jsonl          |   20   |        7        |  13  |   15  |   28  |
| refusals.jsonl           |   15   |        1        |  14  |   12  |   26  |
| humor.jsonl              |   10   |        1        |   9  |   10  |   19  |
| conversation.jsonl (new) |    0   |        0        |   0  |   20  |   20  |
| docker.jsonl (new)       |    0   |        0        |   0  |   20  |   20  |
| fastapi.jsonl            |    4   |        0        |   4  |   16  |   20  |
| git.jsonl                |    5   |        0        |   5  |   15  |   20  |
| linux.jsonl              |    7   |        0        |   7  |   13  |   20  |
| python.jsonl             |    5   |        0        |   5  |   15  |   20  |
| Total                    |   218  |        39       |  179 |  244  |  423  |

## Verified

- `tools/validate_dataset.py` against the full 423-sample set: all valid, no duplicate
  IDs, schema clean.
- `ai/training/build_vocab.py` re-run against it: vocab grew from 94 to 98 characters
  (new punctuation/characters from the expanded English text; the `vocab.json` in this
  zip is the regenerated one — drop it in, no need to rerun build_vocab yourself unless
  you add more data later).

## How to use this

Drop `datasets/train/*.jsonl` and `datasets/tokenizer/vocab.json` into your project,
overwriting the old ones, then retrain. Nothing else needs to change.
