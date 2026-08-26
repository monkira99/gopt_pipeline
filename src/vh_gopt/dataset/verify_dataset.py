#!/usr/bin/env python3
"""Cong chat luong bat buoc truoc khi push dataset len HF.

Kiem tra tren bo 4 file .npz (train / val / test_unseen_speakers / test_unseen_prompts):
  - cau truc tensor: dung mang, dtype, phn trong [0,38] + pad -1, phone_list 39 am vi
  - do phu nhan: phone_valid / word_valid >= nguong, utt acc+total = 100%
  - nhat quan mask: label < 0  =>  weight == 0 (phone/word/utt-fluency)
  - feat phai la feature that (nonzero > 5%) truoc khi push (--allow-empty-feat de test)
  - zero-leakage: id disjoint; text train khong xuat hien o test_unseen_prompts;
    speaker disjoint train/val/test_unseen_speakers (can --manifest de lay user_id)
Exit code 1 neu co ERROR. Bao cao in ra dang JSON.
"""
import argparse
import json
import sys

import numpy as np

from vh_gopt.config import add_config_arg, load_config

REQUIRED_2D = ["phn", "word_id", "phone_label", "phone_weight", "n_vendors",
               "word_acc", "word_weight", "msdd_type", "msdd_sub"]
SPLITS = ["train", "val", "test_unseen_speakers", "test_unseen_prompts"]


def audit_split(z, errors, warns):
    L = int(z["max_len"])
    pad = z["phn"] < 0
    nvp = int((~pad).sum())

    # phone_list gan sat ARPA39
    if len(z["phone_list"]) != 39:
        errors.append(f"phone_list co {len(z['phone_list'])} am vi (mong doi 39)")
    pmin, pmax = z["phn"][~pad].min(), z["phn"][~pad].max()
    if pmin < 0 or pmax > 38:
        errors.append(f"phn vuot khoang [0,38]: [{pmin},{pmax}]")

    # do phu nhan
    cov = {
        "phone": float(((z["phone_label"] >= 0) & ~pad).sum()) / nvp,
        "word": float(((z["word_acc"] >= 0) & ~pad).sum()) / nvp,
    }
    for head in (0, 3):  # accuracy, total phai 100%
        bad = int((z["utt_label"][:, head] < 0).sum())
        if bad:
            errors.append(f"utt_label head {head} co {bad} mau bi mask (mong doi 100%)")
    if float((z["occ"][~pad] == 0).mean()) > 0.999:
        warns.append("occ toan 0 tai vi tri hop le (chua trich occupancy?)")
    return {"L": L, "pad": pad, "nvp": nvp, "cov": cov}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_arg(ap)
    ap.add_argument("--npz-dir", default=None)
    ap.add_argument("--manifest", default=None,
                    help="manifest jsonl goc de kiem tra leakage theo user_id (khuyet nghien: corpus manifest)")
    ap.add_argument("--corpus-dir", default=None)
    ap.add_argument("--min-phone-valid", type=float, default=0.90)
    ap.add_argument("--min-word-valid", type=float, default=0.95)
    ap.add_argument("--allow-empty-feat", action="store_true",
                    help="chap nhan feat toan so 0 (CHI danh cho test noi bo, cam push that)")
    ap.add_argument("--splits", default=",".join(SPLITS),
                    help="danh sach split can kiem tra, phan cach boi dau phay")
    args = ap.parse_args()

    cfg = load_config(args.config)
    npz_dir = args.npz_dir or cfg["out_dir"]
    rep = {"npz_dir": npz_dir, "errors": [], "warnings": [], "splits": {}}

    Z = {}
    for s in [x.strip() for x in args.splits.split(",") if x.strip()]:
        try:
            Z[s] = np.load(f"{npz_dir}/{s}.npz", allow_pickle=False)
        except FileNotFoundError:
            rep["errors"].append(f"Thieu file {s}.npz trong {npz_dir}")
    if rep["errors"]:
        print(json.dumps(rep, indent=2))
        sys.exit(1)

    info = {}
    E, W = rep["errors"], rep["warnings"]
    for s, z in Z.items():
        a = audit_split(z, E, W)
        cov = a["cov"]
        if cov["phone"] < args.min_phone_valid:
            E.append(f"[{s}] phone_valid={cov['phone']*100:.2f}% < {args.min_phone_valid*100:.0f}%")
        if cov["word"] < args.min_word_valid:
            E.append(f"[{s}] word_valid={cov['word']*100:.2f}% < {args.min_word_valid*100:.0f}%")
        # nhat quan mask-weight
        for arr, warr, name in (("phone_label", "phone_weight", "phone"),
                                ("word_acc", "word_weight", "word")):
            v = (z[arr] < 0) & (z[warr] != 0)
            if v.any():
                W.append(f"[{s}] {name}: {int(v.sum())} vi tri masked nhung weight!=0")
        fl = z["utt_label"][:, 2] < 0
        if (z["utt_weight"][fl, 2] != 0).any():
            E.append(f"[{s}] fluency masked nhung utt_weight!=0")
        # feat that hay placeholder
        nz = float((z["feat"] != 0).mean())
        if nz < 0.05 and not args.allow_empty_feat:
            E.append(f"[{s}] feat chi {nz*100:.2f}% phan tu khac 0 -> placeholder, "
                     f"phai pack lai voi acoustic model ({cfg['model_id']}) hoac --allow-empty-feat")
        # msdd
        if not set(np.unique(z["msdd_type"])).issubset({-1, 0, 1, 2}):
            E.append(f"[{s}] msdd_type co gia tri ngoai {{-1,0,1,2}}")
        if z["msdd_sub"].max() > 38:
            E.append(f"[{s}] msdd_sub vuot 38")
        # cat cut bien phai
        hit_last = int((~a["pad"][:, -1]).sum())
        if hit_last:
            W.append(f"[{s}] {hit_last} mau cham cot cuoi max_len={a['L']} (co the bi cat)")
        rep["splits"][s] = {
            "N": int(z["N"]), "max_len": a["L"],
            "phone_valid_pct": round(cov["phone"] * 100, 2),
            "word_valid_pct": round(cov["word"] * 100, 2),
            "feat_nonzero_pct": round(nz * 100, 2),
            "msdd_gold": int((z["msdd_sub"] >= 0).sum()),
            "hit_last_col": hit_last,
        }

    # ---- Zero-leakage ----
    ids = {s: set(map(str, z["ids"])) for s, z in Z.items()}
    txt = {s: set(map(str, z["texts"])) for s, z in Z.items()}
    all_pairs = [("train", "val"), ("train", "test_unseen_speakers"),
                 ("train", "test_unseen_prompts"), ("val", "test_unseen_speakers")]
    pairs = [(a, b) for a, b in all_pairs if a in ids and b in ids]
    for a, b in pairs:
        ov = ids[a] & ids[b]
        if ov:
            E.append(f"Leak ID giua {a} va {b}: {len(ov)}")
    if "train" in txt and "test_unseen_prompts" in txt:
        ov = txt["train"] & txt["test_unseen_prompts"]
        if ov:
            E.append(f"Leak text giua train va test_unseen_prompts: {len(ov)} de bai")

    corpus_dir = args.corpus_dir or cfg["corpus_dir"]
    man_path = args.manifest
    if man_path is None:
        from pathlib import Path
        from vh_gopt.dataset.corpus import CORPUS_LAYOUT
        cand = Path(corpus_dir) / CORPUS_LAYOUT["manifest"]
        man_path = str(cand) if cand.exists() else None
        if man_path is None:
            W.append("Khong tim thay manifest -> bo qua kiem tra leakage theo speaker")
    if man_path:
        users = {}
        for line in open(man_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                k = r.get("ID") or r.get("id") or r.get("audio_id")
                users[k] = r.get("user_id")
        spk = {s: {users.get(i) for i in ids[s]} for s in ids}
        # speaker chi bat buoc disjoint giua cac split unseen-speaker; test_unseen_prompts
        # duoc phep dung lai speaker cua train (kiem tra chung de bao cao, khong phat loi)
        spk_pairs = [(a, b) for a, b in [("train", "val"), ("train", "test_unseen_speakers"),
                                         ("val", "test_unseen_speakers")]
                     if a in spk and b in spk]
        for a, b in spk_pairs:
            ov = spk[a] & spk[b]
            if ov:
                E.append(f"Leak speaker giua {a} va {b}: {len(ov)}")
        rep["speakers"] = {s: len(v) for s, v in spk.items()}

    status = "PASS" if not E else "FAIL"
    rep["status"] = status
    print(json.dumps(rep, indent=2))
    sys.exit(0 if not E else 1)


if __name__ == "__main__":
    main()
