#!/usr/bin/env python3
"""Tao corpus snapshot tu du lieu thoi (chay tren may hien co du lieu).

Gom: manifest + splits + vendor JSON (SpeechAce/SpeechSuper/iFlytek) cua dung cac ID
trong manifest + audio (hardlink, fallback copy) -> thu muc snapshot chuan.
Optionally push snapshot len HF dataset repo (private) de server khac fetch.

Cach dung:
  python -m vh_gopt.dataset.snapshot_corpus \
      --src-root data/vh_alive_2026_scripted --ss-dir cache/ss_vh --ifly-dir cache/iflytek_vh \
      --out data/corpus [--push-repo <user>/gopt-vh-corpus] [--public]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from vh_gopt.config import add_config_arg, load_config
from vh_gopt.dataset.corpus import CORPUS_LAYOUT


def link_or_copy(src, dst):
    if dst.exists():
        return "dup"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_arg(ap)
    ap.add_argument("--src-root", default="data/vh_alive_2026_scripted",
                    help="thu muc goc chua manifest/splits/vendor_eval/audio")
    ap.add_argument("--ss-dir", default="cache/ss_vh")
    ap.add_argument("--ifly-dir", default="cache/iflytek_vh")
    ap.add_argument("--manifest", default=None, help="override (mac dinh <src-root>/manifest.scripted_gold.jsonl)")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--out", default=None, help="thu muc snapshot dau ra (mac dinh cfg corpus_dir)")
    ap.add_argument("--push-repo", default=None, help="<user>/<repo> day snapshot len HF (private)")
    ap.add_argument("--limit", type=int, default=0,
                    help="chi gom N mau dau tien (smoke test; splits se duoc ghi lai cho dung N mau nay)")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or cfg["corpus_dir"])
    src = Path(args.src_root)
    manifest_path = Path(args.manifest or src / CORPUS_LAYOUT["manifest"])
    splits_path = Path(args.splits or src / CORPUS_LAYOUT["splits"])

    rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8") if l.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]
    splits = json.load(open(splits_path, encoding="utf-8"))

    for sub in ("ace_dir", "ss_dir", "ifly_dir", "audio_dir"):
        (out / CORPUS_LAYOUT[sub]).mkdir(parents=True, exist_ok=True)

    stats = {"n_manifest": len(rows), "missing": [], "audio": 0, "ace": 0, "ss": 0, "ifly": 0}
    for r in rows:
        rid = r["ID"]
        # vendor SpeechAce nam trong <src-root>/vendor_eval/<ID>.json
        for key, sdir, dstdir in (("ace", src / CORPUS_LAYOUT["ace_dir"], out / CORPUS_LAYOUT["ace_dir"]),
                                  ("ss", Path(args.ss_dir), out / CORPUS_LAYOUT["ss_dir"]),
                                  ("ifly", Path(args.ifly_dir), out / CORPUS_LAYOUT["ifly_dir"])):
            f = sdir / f"{rid}.json"
            if f.exists():
                link_or_copy(f, dstdir / f"{rid}.json")
                stats[key] += 1
            else:
                stats["missing"].append(f"{key}:{rid}")
        # audio: uu tien basename trong audio_path (tuong doi voi src-root hoac tuyet doi)
        ap_ = r.get("audio_path") or ""
        cands = [src / CORPUS_LAYOUT["audio_dir"] / Path(ap_).name, Path(ap_) if Path(ap_).is_absolute() else None]
        for c in cands:
            if c and c.exists():
                link_or_copy(c, out / CORPUS_LAYOUT["audio_dir"] / c.name)
                stats["audio"] += 1
                break

    shutil.copy2(manifest_path, out / CORPUS_LAYOUT["manifest"])
    shutil.copy2(splits_path, out / CORPUS_LAYOUT["splits"])

    meta = {
        "source_manifest": str(manifest_path), "n_manifest": stats["n_manifest"],
        "counts": {k: stats[k] for k in ("audio", "ace", "ss", "ifly")},
        "n_missing_vendor": len(stats["missing"]),
        "split_sizes": {k: len(v) for k, v in splits.items() if isinstance(v, list)},
    }
    json.dump(meta, open(out / "snapshot_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))
    if stats["missing"]:
        p = out / "snapshot_missing.txt"
        p.write_text("\n".join(stats["missing"]))
        print(f"[WARN] {len(stats['missing'])} file vendor thieu -> {p}")

    if args.push_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        repo = args.push_repo if "/" in args.push_repo else f"{api.whoami()['name']}/{args.push_repo}"
        api.create_repo(repo, repo_type="dataset", private=not args.public, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="dataset")
        print(f"Pushed snapshot -> https://huggingface.co/datasets/{repo} (private={not args.public})")


if __name__ == "__main__":
    main()
