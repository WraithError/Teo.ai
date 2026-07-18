import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.training.train_language_model import save_best_checkpoint


class DummyLayer:
    def __init__(self):
        self.params = {"w": np.ones((2, 2), dtype=np.float32)}


class DummyTokenizer:
    vocab_size = 5


class DummyCfg:
    hidden_size = 16
    lr = 0.001
    batch_size = 2
    max_seq_len = 32
    grad_clip = 1.0


def test_save_best_checkpoint_writes_npz_and_card(tmp_path):
    lstm = DummyLayer()
    dense = DummyLayer()
    cfg = DummyCfg()
    tokenizer = DummyTokenizer()

    output_dir = tmp_path / "checkpoints"
    save_best_checkpoint(
        lstm,
        dense,
        epoch=7,
        train_loss=0.4,
        val_loss=0.2,
        perplexity=1.2,
        tokenizer=tokenizer,
        cfg=cfg,
        path=output_dir,
    )

    weights_path = output_dir / "best_weights.npz"
    card_path = output_dir / "model_card.json"

    assert weights_path.exists()
    assert card_path.exists()

    with np.load(weights_path) as data:
        assert "layer0_w" in data
        assert "layer1_w" in data

    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["saved_epoch"] == 7
    assert card["architecture"]["hidden_size"] == 16
