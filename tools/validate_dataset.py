"""
tools/validate_dataset.py

Run this after every batch of new dialogue samples, before training.
Checks the agreed schema (see datasets/README.md) across every .jsonl
file in datasets/train/, datasets/validation/, datasets/test/.

Usage:
    python tools/validate_dataset.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ["train", "validation", "test"]

REQUIRED_FIELDS = {
    "id": int,
    "source": str,
    "language": str,
    "category": str,
    "difficulty": str,
    "emotion": str,
    "tags": list,
    "user": str,
    "assistant": str,
}

VALID_SOURCE = {"human", "generated", "corrected", "augmented"}
VALID_LANGUAGE = {"en", "uz", "ru", "mixed"}
VALID_DIFFICULTY = {"easy", "medium", "hard", "expert"}


def validate_split(split: str) -> tuple[list[str], int, Counter, set]:
    split_dir = ROOT / "datasets" / split
    errors: list[str] = []
    count = 0
    categories: Counter = Counter()
    ids: set = set()

    if not split_dir.exists():
        return errors, count, categories, ids

    for fpath in sorted(split_dir.glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                count += 1
                loc = f"{fpath.name}:{line_num}"

                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"{loc} — invalid JSON: {e}")
                    continue

                for field, expected_type in REQUIRED_FIELDS.items():
                    if field not in d:
                        errors.append(f"{loc} — missing field '{field}'")
                    elif not isinstance(d[field], expected_type):
                        errors.append(
                            f"{loc} — field '{field}' should be {expected_type.__name__}, "
                            f"got {type(d[field]).__name__}"
                        )

                if "id" in d:
                    if d["id"] in ids:
                        errors.append(f"{loc} — duplicate id {d['id']}")
                    ids.add(d["id"])

                if "source" in d and d["source"] not in VALID_SOURCE:
                    errors.append(f"{loc} — invalid source '{d['source']}'")
                if "language" in d and d["language"] not in VALID_LANGUAGE:
                    errors.append(f"{loc} — invalid language '{d['language']}'")
                if "difficulty" in d and d["difficulty"] not in VALID_DIFFICULTY:
                    errors.append(f"{loc} — invalid difficulty '{d['difficulty']}'")

                if "user" in d and isinstance(d["user"], str) and not d["user"].strip():
                    errors.append(f"{loc} — empty 'user' field")
                if "assistant" in d and isinstance(d["assistant"], str) and not d["assistant"].strip():
                    errors.append(f"{loc} — empty 'assistant' field")

                if "category" in d:
                    categories[d["category"]] += 1

    return errors, count, categories, ids


def main() -> int:
    all_errors: list[str] = []
    total = 0
    all_categories: Counter = Counter()
    all_ids: set = set()

    for split in SPLITS:
        errors, count, categories, ids = validate_split(split)
        overlap = all_ids & ids
        if overlap:
            all_errors.append(
                f"ID collision ACROSS splits: {sorted(overlap)[:10]}"
                f"{' ...' if len(overlap) > 10 else ''}"
            )
        all_ids |= ids
        all_errors.extend(errors)
        total += count
        all_categories.update(categories)

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    print(f"  Total samples checked: {total}")
    print(f"  Unique IDs:            {len(all_ids)}")
    print()
    print("  By category:")
    for cat, n in sorted(all_categories.items(), key=lambda x: -x[1]):
        print(f"    {cat:<14} {n}")
    print()

    if all_errors:
        print(f"  {len(all_errors)} ERROR(S):")
        for e in all_errors[:50]:
            print(f"    ✗ {e}")
        if len(all_errors) > 50:
            print(f"    ... and {len(all_errors) - 50} more")
        print()
        print("  FAILED — fix the above before training.")
        return 1

    print("  All samples valid. Schema clean, no duplicate IDs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
