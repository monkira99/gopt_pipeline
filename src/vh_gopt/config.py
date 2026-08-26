"""Pipeline defaults + optional JSON config override.

Uu tien: CLI flag > config file > DEFAULTS.
"""
import json
import os

DEFAULTS = {
    # Thu muc snapshot du lieu dau vao (xem vh_gopt.dataset.corpus.CORPUS_LAYOUT)
    "corpus_dir": "data/corpus",
    # Noi xuat 4 file .npz trung gian
    "out_dir": "data/gopt_vh_scripted_gold",
    # Acoustic model cho Stage-1 CTC-GOP feature (80-d)
    "model_id": "KoelLabs/xlsr-english-01",
    "device": "cpu",
    "max_len": 150,
    # Repo HF chua RAW snapshot (vendor json + manifest + audio) - private
    "hf_corpus_repo": None,
    # Repo HF dang datasets.DatasetDict sau khi push - private
    "hf_dataset_repo": None,
}


def load_config(path=None):
    cfg = dict(DEFAULTS)
    candidates = [path, os.environ.get("VH_GOPT_CONFIG"), "configs/vh_gold.json"]
    for c in candidates:
        if c and os.path.exists(c):
            cfg.update(json.load(open(c, encoding="utf-8")))
            break
    return cfg


def add_config_arg(ap):
    ap.add_argument("--config", default=None, help="JSON config (mac dinh configs/vh_gold.json)")
