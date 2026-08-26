#!/usr/bin/env python3
"""Build GOLD dataset dang Arrow (datasets.DatasetDict) de push len HuggingFace.

Moi row = 1 audio read-aloud gold (6,361 cau) + NHAN day du Stage 2 / MSDD da
dong thuan 3 vendor (SpeechAce ∩ SpeechSuper ∩ iFlytek), thang [0,100].

TAT CA thong tin phu tro deu la cot trong tung row (khong co file metadata
rieng): phone_list (ARPA39 id -> ten am vi), utt_heads, word_heads, scale,
max_len. Server khac chi can `load_dataset()` la co du lieu hoan chinh.

KHONG chua GOP feature (80-d KoelLabs) va occupancy: se extract sau tren may
manh hon TRUC TIEP TU COT AUDIO roi day len repo nhu version moi.

Nguon dau vao:
  --npz-dir  : 4 file .npz trung gian tu pack_stage2 (nhan da qua verify)
  --src-root : goc VuiHoc chua audio/<ID>.mp3 + manifest.scripted_gold.jsonl

Dung:
  ./run.sh build                                   # build local -> data/gopt_vh_gold_arrow
  ./run.sh build --repo <org>/gopt-vh-gold         # build + push private len HF
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from vh_gopt.config import add_config_arg, load_config

SPLITS = ["train", "val", "test_unseen_speakers", "test_unseen_prompts"]
LABEL_COLS = ["phn", "phone_label", "phone_weight", "n_vendors",
              "word_id", "word_acc", "word_weight",
              "msdd_type", "msdd_sub"]

SPLIT_ROLE = {
    "train": "training",
    "val": "checkpoint selection (unseen speakers)",
    "test_unseen_speakers": "generalization to unseen speakers",
    "test_unseen_prompts": "generalization to unseen prompts (7 new questions)",
}


def build_features(L):
    from datasets import Audio, Sequence, Value
    seq = lambda dt, n=L: Sequence(Value(dt), length=n)  # noqa: E731
    return {
        "id": Value("string"),
        "audio": Audio(),
        "text": Value("string"),
        "user_id": Value("int64"),
        "question_id": Value("int64"),
        "duration_sec": Value("float32"),
        "split_role": Value("string"),
        # ---- hang so cap dataset, lap o tung row de load_dataset tu-sufficiency ----
        "phone_list": Sequence(Value("string"), length=39),
        "utt_heads": Sequence(Value("string"), length=4),
        "word_heads": Sequence(Value("string"), length=1),
        "scale": Value("string"),
        "max_len": Value("int32"),
        # ---- nhan ----
        "phn": seq("int16"),
        "phone_label": seq("float32"),
        "phone_weight": seq("float32"),
        "n_vendors": seq("uint8"),
        "word_id": seq("int16"),
        "word_acc": seq("float32"),
        "word_weight": seq("float32"),
        "msdd_type": seq("int16"),
        "msdd_sub": seq("int16"),
        "utt_label": Sequence(Value("float32"), length=4),
        "utt_weight": Sequence(Value("float32"), length=4),
        "utt_nv": Sequence(Value("uint8"), length=4),
    }


def make_card(repo, stats):
    rows = "\n".join(
        f"| {s} | {v['N']:,} | {v['speakers']:,} | {v['phone_valid_pct']}% | {v['word_valid_pct']}% |"
        for s, v in stats.items())
    return f"""---
language:
- en
pretty_name: "VuiHoc GOPT Gold (read-aloud, 3-vendor consensus labels)"
tags:
- goodness-of-pronunciation
- pronunciation-assessment
- msdd
- speech
size_categories:
- 1K<n<10K
---

# VuiHoc GOPT Gold — audio + consensus labels (Arrow)

6,361 cau IELTS read-aloud thuan sach (~35h) tu he thong VuiHoc, moi row gom
**audio + nhan** dong thuan 3 vendor (SpeechAce, SpeechSuper, iFlytek ISE),
thang nghiep vu [0.0, 100.0], chia 4 split **zero-leakage**.

## Splits

| Split | Mau | Speakers | Phone valid | Word valid |
|---|---|---|---|---|
{rows}

- Speaker khong giao nhau giua train / val / test_unseen_speakers.
- test_unseen_prompts dung 7 de bai hoan toan moi (speaker co the trung train).

## Schema (moi row)

| Cot | Kieu | Mo ta |
|---|---|---|
| `id` | string | ID audio VuiHoc |
| `audio` | Audio | mp3 goc |
| `text` | string | de bai canonical |
| `user_id` / `question_id` | int64 | nguoi hoc / de bai |
| `duration_sec` | float32 | do dai audio |
| `split_role` | string | y nghia kiem thu cua split |
| `phone_list` | [39] string | bang tra ARPA39: index trong `phn`/`msdd_sub` -> ten am vi |
| `utt_heads` | [4] string | thu tu head utterance: accuracy, completeness, fluency, total |
| `word_heads` | [1] string | head word: accuracy |
| `scale` | string | thang diem: 0-100 |
| `max_len` | int32 | do dai pad cua cac sequence |
| `phn` | [{stats[next(iter(stats))]['L']}] int16 | id ARPA39 canonical, pad = -1 |
| `phone_label` / `phone_weight` | float32 | diem am vi dong thuan [0,100] / trong so tin cay [0,1]; mask -1.0 |
| `n_vendors` | uint8 | so voter tai vi tri (0-3) |
| `word_id` | int16 | chi so tu (trong text) de word-pooling, pad = -1 |
| `word_acc` / `word_weight` | float32 | word accuracy native vendor / trong so; mask -1.0 |
| `msdd_type` | int16 | 0=OK, 1=Substitution, 2=Deletion, -1=Mask |
| `msdd_sub` | int16 | id am vi thay the (dong thuan Ace∩SS), mask = -1 |
| `utt_label` / `utt_weight` | [[4]] float32 | (accuracy, completeness, fluency, total) / trong so |
| `utt_nv` | [[4]] uint8 | so voter cua tung head |

## Quy uoc quan trong

- `-1` / `-1.0` la MASK (khong co nhan), KHONG phai diem so — loc bang `(label >= 0)`.
- Phone: Median-of-3 vendor sau chuan hoa z-score ve khong gian SpeechSuper;
  2 vendor chi nhan khi |dz| <= 1; nv <= 1 hoac bat dong -> mask.
- Fluency bi mask tren mot phan mau (2 vendor cham lech > 1 sigma).

## GOP feature (se cap nhat)

Cot `feat` (CTC-GOP 80-d KoelLabs/xlsr-english-01 + occupancy) CHUA co trong
version nay — se extract tu cot `audio` va day them len repo sau.

## Load

```python
from datasets import load_dataset
ds = load_dataset("{repo or "<org>/gopt-vh-gold"}")
ex = ds["train"][0]
ex["phone_list"], ex["utt_heads"]   # bang tra nam ngay trong row
```"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_arg(ap)
    ap.add_argument("--npz-dir", default=None, help="thu muc 4 npz nhan (mac dinh cfg out_dir)")
    ap.add_argument("--src-root", default="data/vh_alive_2026_scripted",
                    help="goc chua audio/<ID>.mp3 + manifest.scripted_gold.jsonl")
    ap.add_argument("--out", default=None, help="thu muc save_to_disk (mac dinh data/gopt_vh_gold_arrow)")
    ap.add_argument("--repo", default=None, help="<org>/<name>: push len HF (private truoc khi --public)")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--splits", default=",".join(SPLITS))
    args = ap.parse_args()

    cfg = load_config(args.config)
    npz_dir = args.npz_dir or cfg["out_dir"]
    out_dir = args.out or "data/gopt_vh_gold_arrow"
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    from datasets import Dataset, DatasetDict, Features

    src = Path(args.src_root)
    manifest = {}
    for line in open(src / "manifest.scripted_gold.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            manifest[r["ID"]] = r

    dd, stats, z0 = {}, {}, None
    for s in splits:
        z = np.load(os.path.join(npz_dir, f"{s}.npz"), allow_pickle=False)
        if s == "train":
            z0 = z
        N, L = int(z["N"]), int(z["max_len"])
        feats = build_features(L)
        pad = z["phn"] < 0
        data = {k: [] for k in feats}
        missing_audio = []

        const = {
            "phone_list": [str(x) for x in z["phone_list"]],
            "utt_heads": [str(x) for x in z["utt_heads"]],
            "word_heads": [str(x) for x in z["word_heads"]],
            "scale": str(z["scale"]),
            "max_len": L,
        }
        for i in range(N):
            rid = str(z["ids"][i])
            m = manifest.get(rid)
            apath = None
            if m and m.get("audio_path"):
                cand = Path(m["audio_path"])
                if cand.is_absolute():
                    apath = cand if cand.exists() else None
                else:
                    cand = src / "audio" / os.path.basename(m["audio_path"])
                    apath = cand if cand.exists() else None
            if apath is None:
                missing_audio.append(rid)
                continue
            row = {
                "id": rid,
                "audio": {"path": str(apath.resolve()), "bytes": None},
                "text": str(z["texts"][i]),
                "user_id": int(m["user_id"]) if m else -1,
                "question_id": int(m.get("question_id") or -1),
                "duration_sec": float(m.get("duration_sec") or -1.0),
                "split_role": SPLIT_ROLE.get(s, ""),
                **const,
            }
            for k in LABEL_COLS:
                row[k] = z[k][i].tolist()
            for k in ("utt_label", "utt_weight", "utt_nv"):
                row[k] = z[k][i].tolist()
            for k, v in row.items():
                data[k].append(v)

        if missing_audio:
            raise SystemExit(f"[{s}] thieu audio cho {len(missing_audio)} ID, vd: {missing_audio[:5]}")

        ds = Dataset.from_dict(data, features=Features(feats))
        pl = np.asarray(ds["phone_label"]); wa = np.asarray(ds["word_acc"])
        nvp = int((~pad).sum())
        stats[s] = {"N": N, "L": L,
                    "speakers": len({x for x in ds["user_id"]}),
                    "phone_valid_pct": round(float(((pl >= 0) & ~pad).sum()) / nvp * 100, 2),
                    "word_valid_pct": round(float(((wa >= 0) & ~pad).sum()) / nvp * 100, 2)}
        dd[s] = ds
        print(f"[{s}] {N:,} rows | speakers={stats[s]['speakers']} "
              f"| phone_valid={stats[s]['phone_valid_pct']}% word_valid={stats[s]['word_valid_pct']}%")

    dsdict = DatasetDict(dd)
    if not args.repo:
        dsdict.save_to_disk(out_dir)
        print(json.dumps(stats, indent=2))
        print(f"\nDa build local -> {out_dir}. Push bang: ./run.sh build --repo <org>/<name>")
        return

    repo = args.repo
    card = make_card(repo, stats)
    private = not args.public
    dsdict.push_to_hub(repo, private=private)
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    print(f"Pushed -> https://huggingface.co/datasets/{repo} (private={private})")


if __name__ == "__main__":
    main()
