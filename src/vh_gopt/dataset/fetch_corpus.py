#!/usr/bin/env python3
"""Tai corpus snapshot tu HF dataset repo ve server build.

Cach dung:
  python -m vh_gopt.dataset.fetch_corpus --repo <user>/gopt-vh-corpus --corpus-dir data/corpus
Yeu cau HF_TOKEN (env) neu repo private.
"""
import argparse

from vh_gopt.config import add_config_arg, load_config
from vh_gopt.dataset.corpus import check_layout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_arg(ap)
    ap.add_argument("--repo", default=None, help="<user>/<repo> (mac dinh cfg hf_corpus_repo)")
    ap.add_argument("--corpus-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    repo = args.repo or cfg.get("hf_corpus_repo")
    if not repo:
        raise SystemExit("Thieu --repo hoac hf_corpus_repo trong config.")
    corpus_dir = args.corpus_dir or cfg["corpus_dir"]

    from huggingface_hub import snapshot_download
    print(f"Tai snapshot {repo} -> {corpus_dir} ...")
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=corpus_dir)

    missing = check_layout(corpus_dir)
    if missing:
        raise SystemExit(f"Snapshot thieu thanh phan: {missing}")
    print("Snapshot day du. San sang cho `python -m vh_gopt.dataset.pack_stage2`.")


if __name__ == "__main__":
    main()
