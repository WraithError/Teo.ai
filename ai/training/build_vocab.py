"""
ai/training/build_vocab.py

Scans every .jsonl file in datasets/train/ and builds the character
vocabulary from every user and assistant string it finds.
Saves the result to datasets/tokenizer/vocab.json.

Run once before your first training session.
Re-run if you add samples that contain characters not seen before
(unusual, but possible when adding Uzbek text with new symbols).

Usage:
    cd Teo.AI
    python ai/training/build_vocab.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "ai" / "tokenizer"))

from tokenizer import CharTokenizer

DATASET_DIR = ROOT / "datasets" / "train"
VOCAB_PATH  = ROOT / "datasets" / "tokenizer" / "vocab.json"


def main() -> None:
    texts: list[str] = []
    sample_count = 0

    if not DATASET_DIR.exists() or not any(DATASET_DIR.glob("*.jsonl")):
        print(f"ERROR: No .jsonl files found in {DATASET_DIR}")
        sys.exit(1)

    for fpath in sorted(DATASET_DIR.glob("*.jsonl")):
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    texts.append(sample["user"])
                    texts.append(sample["assistant"])
                    sample_count += 1
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"WARNING: skipping bad line in {fpath.name}: {e}")

    if not texts:
        print("ERROR: No usable samples found.")
        sys.exit(1)

    print(f"Scanned {sample_count} samples from {DATASET_DIR}")
    tokenizer = CharTokenizer.build_from_texts(texts)

    tokenizer.save(str(VOCAB_PATH))
    print(f"Vocab size : {tokenizer.vocab_size}")
    print(f"Saved to   : {VOCAB_PATH}")

    # Quick sanity check
    test = "Hello! Accepted. Let me check that code."
    encoded = tokenizer.encode_chars(test)
    decoded = tokenizer.decode(encoded, strip_special=False)
    assert decoded == test, f"Encode/decode roundtrip failed: {decoded!r}"
    print("Roundtrip test : PASS")


if __name__ == "__main__":
    main()
