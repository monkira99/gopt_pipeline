"""Corpus snapshot layout + audio resolution.

Layout chuan cua mot snapshot (tren server build):
  <corpus_dir>/
    manifest.scripted_gold.jsonl   # ID, user_id, target_text, audio_path, ...
    splits_scripted_gold.json      # {"train": [...], "val": [...], ...}
    vendor_eval/<ID>.json          # SpeechAce (LMS)
    ss/<ID>.json                   # SpeechSuper (API)
    iflytek/<ID>.json              # iFlytek ISE (WebSocket)
    audio/<ID>.<ext>               # audio thuc te (mp3/wav)
"""
import json
from pathlib import Path

CORPUS_LAYOUT = {
    "manifest": "manifest.scripted_gold.jsonl",
    "splits": "splits_scripted_gold.json",
    "ace_dir": "vendor_eval",
    "ss_dir": "ss",
    "ifly_dir": "iflytek",
    "audio_dir": "audio",
}


def corpus_paths(corpus_dir):
    root = Path(corpus_dir)
    return {k: root / v for k, v in CORPUS_LAYOUT.items()}


def load_manifest(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["ID"]] = r
    return rows


def resolve_audio(corpus_root, row):
    """Tim file audio cho mot row manifest: uu tien <corpus>/audio/<basename>,
    roi den audio_path tuyet doi / tuong doi nhu trong manifest goc."""
    ap = row.get("audio_path") or ""
    base = Path(ap).name
    cands = [Path(corpus_root) / CORPUS_LAYOUT["audio_dir"] / base]
    if ap:
        p = Path(ap)
        if p.is_absolute():
            cands.append(p)
        else:
            cands.append(Path(corpus_root) / p)
    for c in cands:
        if c.exists():
            return c
    return None


def check_layout(corpus_dir):
    """Return list loi con thieu trong layout."""
    missing = []
    p = corpus_paths(corpus_dir)
    for k in ("manifest", "splits"):
        if not p[k].exists():
            missing.append(str(p[k]))
    for k in ("ace_dir", "ss_dir", "ifly_dir", "audio_dir"):
        if not p[k].is_dir():
            missing.append(str(p[k]) + "/")
    return missing
