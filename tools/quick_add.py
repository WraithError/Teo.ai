"""
tools/quick_add.py

Interactive data entry for Teo training samples.
Auto-assigns IDs, validates the entry, appends to the right file.

Usage:
    cd Teo.AI
    python tools/quick_add.py

Commands inside the tool:
    Enter a message   → start a new sample
    skip              → skip optional field (uses default)
    done / exit       → quit
    list              → show current counts per file
"""

import json
import sys
import glob
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "datasets" / "train"

CATEGORIES = [
    "coding", "debugging", "deployment", "greeting",
    "teaching", "security", "personality", "refusals",
    "humor", "linux", "git", "python", "fastapi",
    "docker", "conversation", "project", "random",
]

DIFFICULTIES = ["easy", "medium", "hard", "expert"]
EMOTIONS     = [
    "neutral", "frustrated", "curious", "excited", "calm",
    "serious", "confused", "playful", "panicked", "tired",
    "happy", "motivated", "firm", "encouraging",
]
LANGUAGES    = ["en", "uz", "ru", "mixed"]


def get_next_id() -> int:
    ids = []
    for f in TRAIN_DIR.glob("*.jsonl"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        ids.append(json.loads(line)["id"])
                    except Exception:
                        pass
    return (max(ids) + 1) if ids else 1


def show_counts():
    print("\n  Current sample counts:")
    total = 0
    for f in sorted(TRAIN_DIR.glob("*.jsonl")):
        n = sum(1 for l in open(f, encoding="utf-8") if l.strip())
        print(f"    {f.stem:<16} {n}")
        total += n
    print(f"    {'TOTAL':<16} {total}\n")


def pick(prompt: str, options: list[str], default: str = "") -> str:
    short = [o[:3] for o in options]
    print(f"  {prompt}")
    for i, o in enumerate(options):
        print(f"    {i+1:2}. {o}")
    while True:
        raw = input(f"  → [1-{len(options)}] (default: {default or options[0]}): ").strip()
        if not raw:
            return default or options[0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        # allow typing the value directly
        if raw in options:
            return raw
        print("  Invalid. Try again.")


def get_tags() -> list[str]:
    raw = input("  Tags (comma-separated, e.g. python,fastapi): ").strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def write_sample(sample: dict) -> Path:
    cat = sample["category"]
    fname = TRAIN_DIR / f"{cat}.jsonl"
    with open(fname, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return fname


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TEO DATA ENTRY  (type 'list' to see counts, 'done' to quit)")
    print("=" * 60)

    session_count = 0

    while True:
        print()
        cmd = input("User message (or 'list'/'done'): ").strip()

        if not cmd:
            continue
        if cmd.lower() in {"done", "exit", "quit"}:
            break
        if cmd.lower() == "list":
            show_counts()
            continue

        user_msg = cmd

        print()
        assistant_msg = input("Teo response: ").strip()
        if not assistant_msg:
            print("  Skipped (empty response).")
            continue

        print()
        category   = pick("Category?",   CATEGORIES,   "coding")
        difficulty = pick("Difficulty?", DIFFICULTIES, "medium")
        emotion    = pick("User emotion?", EMOTIONS,   "neutral")
        language   = pick("Language?",   LANGUAGES,    "en")
        tags       = get_tags()

        next_id = get_next_id()
        sample = {
            "id":         next_id,
            "source":     "human",
            "language":   language,
            "category":   category,
            "difficulty": difficulty,
            "emotion":    emotion,
            "tags":       tags,
            "user":       user_msg,
            "assistant":  assistant_msg,
        }

        print()
        print("  Preview:")
        print(f"    #{next_id} [{category}/{difficulty}/{emotion}]")
        print(f"    U: {user_msg[:80]}{'...' if len(user_msg)>80 else ''}")
        print(f"    T: {assistant_msg[:80]}{'...' if len(assistant_msg)>80 else ''}")
        confirm = input("  Save? [Y/n]: ").strip().lower()
        if confirm in {"", "y", "yes"}:
            fpath = write_sample(sample)
            print(f"  Saved → {fpath.name}  (id={next_id})")
            session_count += 1
        else:
            print("  Discarded.")

    print()
    print(f"Session complete. {session_count} sample(s) added.")
    show_counts()


if __name__ == "__main__":
    main()
