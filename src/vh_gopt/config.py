"""Pipeline config loader & saver (supports both YAML and JSON).

Hierarchy: CLI flags > Config File (YAML/JSON) > Defaults.
"""
import json
import os
from pathlib import Path
import yaml

DEFAULTS = {
    # Thu muc snapshot du lieu dau vao
    "corpus_dir": "data/corpus",
    # Noi xuat 4 file .npz trung gian
    "out_dir": "data/gopt_vh_scripted_gold",
    # Acoustic model cho Stage-1 CTC-GOP feature (80-d)
    "model_id": "KoelLabs/xlsr-english-01",
    "device": "cpu",
    "max_len": 150,
    "hf_corpus_repo": None,
    "hf_dataset_repo": "tiennguyenbnbk/gopt-vh-gold",
}


def deep_merge(base, update):
    """Deep merge two dictionaries."""
    res = dict(base)
    for k, v in update.items():
        if k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = deep_merge(res[k], v)
        else:
            res[k] = v
    return res


def load_config_file(path):
    """Load config from YAML or JSON file."""
    p = str(path)
    if p.endswith(".yaml") or p.endswith(".yml"):
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    elif p.endswith(".json"):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    else:
        # Try YAML first, then JSON
        try:
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f) or {}


def save_config(cfg, out_path):
    """Save config dict to YAML or JSON file."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if str(p).endswith(".json"):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    else:
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def load_config(path=None):
    cfg = dict(DEFAULTS)
    candidates = [path, os.environ.get("VH_GOPT_CONFIG"), "configs/vh_gold.json"]
    for c in candidates:
        if c and os.path.exists(c):
            f_cfg = load_config_file(c)
            cfg = deep_merge(cfg, f_cfg)
            break
    return cfg


def add_config_arg(ap):
    ap.add_argument("--config", default=None, help="Đường dẫn file cấu hình YAML/JSON (ví dụ: configs/stage2/baseline_wavlm32.yaml)")
