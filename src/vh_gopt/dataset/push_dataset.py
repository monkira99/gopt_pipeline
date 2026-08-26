#!/usr/bin/env python3
"""Day dataset len HuggingFace Hub o dang datasets.DatasetDict (Arrow).

Nguon: 4 file .npz trung gian trong --npz-dir (san pham cua pack_stage2).
Moi split tro thanh 1 Dataset; moi mau = 1 row chua day du tensor phone/word/utt/
MSDD + GOP feature. Nguoi tieu thu tai lai bang:

    from datasets import load_dataset
    ds = load_dataset("<user>/<repo>")   # ds["train"], ds["val"], ...

Yeu cau verify_dataset PASS truoc khi goi that (--dry-run de kiem tra conversion).
"""
import argparse
import json
import os
import tempfile

import numpy as np

from vh_gopt.config import add_config_arg, load_config

SPLITS = ["train", "val", "test_unseen_speakers", "test_unseen_prompts"]


def build_features(L, D):
    from datasets import Array2D, Sequence, Value
    seq = lambda dt: Sequence(Value(dt), length=L)  # noqa: E731
    return {
        "id": Value("string"),
        "text": Value("string"),
        "feat": Array2D(shape=(L, D), dtype="float32"),
        "occ": seq("float32"),
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


def split_to_dataset(path):
    """npz -> datasets.Dataset. -1/-1.0 giu nguyen lam mask."""
    from datasets import Dataset, Features
    z = np.load(path, allow_pickle=False)
    N, L = int(z["N"]), int(z["max_len"])
    D = z["feat"].shape[-1]
    feats = build_features(L, D)
    cols = {k: z[k] for k in feats if k not in ("id", "text")}
    data = {"id": [str(x) for x in z["ids"]], "text": [str(x) for x in z["texts"]]}
    for k, arr in cols.items():
        data[k] = list(arr) if arr.ndim > 1 else arr.tolist()
    ds = Dataset.from_dict(data, features=Features(feats))
    meta = {
        "phone_list": [str(x) for x in z["phone_list"]],
        "utt_heads": [str(x) for x in z["utt_heads"]],
        "word_heads": [str(x) for x in z["word_heads"]],
        "max_len": L, "gop_dim": D,
        "scale": str(z["scale"]),
        "mask_convention": "-1 (int) / -1.0 (float): vi tri hoac head khong co nhan; "
                           "phone_weight/word_weight/utt_weight trong [0,1] theo do tin cay consensus",
    }
    return ds, meta


def make_card(repo, rep_splits, meta):
    rows = "\n".join(
        f"| {s} | {v['N']:,} | {v['msdd_gold']:,} | {v['phone_valid_pct']}% |"
        for s, v in rep_splits.items())
    schema = "\n".join(
        f"| `{k}` | {v} |" for k, v in [
            ("id", "ID audio VuiHoc"), ("text", "de bai canonical"),
            ("feat", f"[{meta['max_len']},{meta['gop_dim']}] CTC-GOP KoelLabs (log-posterior ratio + norm)"),
            ("occ", f"[{meta['max_len']}] soft occupancy ranh gioi am vi"),
            ("phn", f"[{meta['max_len']}] ARPA39 id, pad -1"),
            ("phone_label / phone_weight", f"[{meta['max_len']}] diem dong thuan 3 vendor [0,100] / trong so [0,1], mask -1.0"),
            ("n_vendors", f"[{meta['max_len']}] so voter tai vi tri"),
            ("word_id", f"[{meta['max_len']}] chi so tu de word-pooling, pad -1"),
            ("word_acc / word_weight", f"[{meta['max_len']}] word accuracy native vendor / trong so"),
            ("msdd_type", f"[{meta['max_len']}] 0=OK 1=Sub 2=Del -1=Mask"),
            ("msdd_sub", f"[{meta['max_len']}] id am vi thay the (dong thuan Ace∩SS), mask -1"),
            ("utt_label / utt_weight / utt_nv", "[4] (accuracy, completeness, fluency, total) / trong so / so voter"),
        ])
    return f"""---
language:
- en
pretty_name: "VuiHoc GOPT Scripted Gold (3-vendor consensus)"
tags:
- goodness-of-pronunciation
- pronunciation-assessment
- msdd
- speech
size_categories:
- 1K<n<10K
---

# VuiHoc GOPT & MSDD — Scripted Gold (consensus 3 vendor)

Dataset GoP (Goodness of Pronunciation) cho mo hinh GOPT Stage 2 + nhan MSDD Stage 1,
dong thuan 3 vendor thuong mai (SpeechAce, SpeechSuper, iFlytek ISE) tren
6,361 cau read-aloud scripted thuan sach (~35h), thang nghiep vu [0.0, 100.0].

## Zero-leakage splits

| Split | Mau | MSDD gold | Phone valid |
|---|---|---|---|
{rows}

- `train` / `val` / `test_unseen_speakers`: speaker hoan toan khong giao nhau.
- `test_unseen_prompts`: 7 de bai hoan toan moi.

## Schema

{schema}

## Quy tac consensus (tom tat)

- Phone: Median-of-3 (w=1/(1+Delta)), mean-of-2 neu |dz|<=1 (w=0.6/(1+dz)), mask neu nv<=1 hoac lech.
- Word/Utterance: median/mean qua cac vendor da chuan hoa z-score ve khong gian SpeechSuper;
  fluency mask khi 2 vendor bat dong (|dz|>1).
- Chi tiet: pipeline `vh_gopt.dataset.pack_stage2` (repo `vh-gopt`).

## Load

```python
from datasets import load_dataset
ds = load_dataset("{repo}")
ex = ds["train"][0]
```

Luu y: gia tri -1 / -1.0 la MASK (khong phai diem so); bo qua chung qua mask
`(label >= 0)` khi tinh loss/metric.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_arg(ap)
    ap.add_argument("--npz-dir", default=None)
    ap.add_argument("--repo", default=None, help="<user>/<name> (mac dinh cfg hf_dataset_repo)")
    ap.add_argument("--public", action="store_true", help="publish cong khai (mac dinh private)")
    ap.add_argument("--dry-run", action="store_true",
                    help="chi convert + save_to_disk thu muc tam, khong push mang")
    ap.add_argument("--splits", default=",".join(SPLITS),
                    help="danh sach split se push (mac dinh ca 4)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    npz_dir = args.npz_dir or cfg["out_dir"]
    repo = args.repo or cfg.get("hf_dataset_repo")
    if not repo and not args.dry_run:
        raise SystemExit("Thieu --repo hoac hf_dataset_repo trong config.")

    from datasets import DatasetDict
    dd, metas = {}, {}
    for s in [x.strip() for x in args.splits.split(",") if x.strip()]:
        p = os.path.join(npz_dir, f"{s}.npz")
        print(f"Convert {p} ...")
        dd[s], metas[s] = split_to_dataset(p)
        print(f"  {s}: {len(dd[s]):,} rows")
    dsdict = DatasetDict(dd)

    if args.dry_run:
        tmp = tempfile.mkdtemp(prefix="vh_gopt_dryrun_")
        out = os.path.join(tmp, "datasetdict")
        dsdict.save_to_disk(out)
        json.dump(metas["train"], open(os.path.join(tmp, "metadata.json"), "w"), indent=2)
        print(f"[DRY-RUN] Convert OK -> {out}")
        return

    private = not args.public

    def split_stats(d):
        phn = np.asarray(d["phn"]); pl = np.asarray(d["phone_label"])
        sub = np.asarray(d["msdd_sub"]); pad = phn < 0
        nvp = int((~pad).sum())
        return {"N": phn.shape[0], "msdd_gold": int((sub >= 0).sum()),
                "phone_valid_pct": round(float(((pl >= 0) & ~pad).sum()) / nvp * 100, 2) if nvp else 0.0}

    card = make_card(repo, {s: split_stats(d) for s, d in dd.items()}, metas["train"])
    dsdict.push_to_hub(repo, private=private)
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(path_or_fileobj=json.dumps(metas["train"], indent=2).encode(),
                    path_in_repo="metadata.json", repo_id=repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    print(f"Pushed -> https://huggingface.co/datasets/{repo} (private={private})")


if __name__ == "__main__":
    main()
