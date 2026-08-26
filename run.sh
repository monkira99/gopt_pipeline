#!/usr/bin/env bash
# Dispatcher moi trao cho vh-gopt pipeline.
#   ./run.sh install                    # pip install -e . (chay 1 lan)
#   ./run.sh snapshot [flags...]        # (may co du lieu thoi) tao + push corpus snapshot len HF
#   ./run.sh fetch    [flags...]        # (server build) tai corpus snapshot tu HF
#   ./run.sh pack     [flags...]        # corpus -> 4 npz (mac dinh trich GOP 80-d; --skip-gop de nhan-only)
#   ./run.sh verify   [flags...]        # cong chat luong truoc khi push
#   ./run.sh push     [flags...]        # npz -> datasets.DatasetDict -> HF Hub (--dry-run de thu conversion)
#   ./run.sh build    [flags...]        # npz nhan + audio -> GOLD Arrow (audio+labels, load_dataset duoc); --repo de push
#   ./run.sh extract  [flags...]        # (server GPU) trich day du Feature: KoelLabs 80-d + Prosody 8-d + WavLM 1024-d -> 4 .npz
#   ./run.sh train    [flags...]        # train GOPT/HIA tren npz trung gian
set -euo pipefail
# Giu cwd cua nguoi goi: cac path tuong doi (data/, cache/) tinh theo noi chay.
# Uu tien .venv trong thu muc module, roi .venv thu muc cha (layout monorepo); PY=... de override.
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${PY:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
  elif [ -x "$ROOT/../.venv/bin/python" ]; then PY="$ROOT/../.venv/bin/python"
  else PY="python3"; fi
fi

cmd="${1:-}"; shift || true
case "$cmd" in
  install) "$PY" -m pip install -e . ;;
  snapshot) "$PY" -m vh_gopt.dataset.snapshot_corpus "$@" ;;
  fetch)    "$PY" -m vh_gopt.dataset.fetch_corpus "$@" ;;
  pack)     "$PY" -m vh_gopt.dataset.pack_stage2 "$@" ;;
  verify)   "$PY" -m vh_gopt.dataset.verify_dataset "$@" ;;
  push)     "$PY" -m vh_gopt.dataset.push_dataset "$@" ;;
  build)    "$PY" -m vh_gopt.dataset.build_gold_arrow "$@" ;;
  extract)  "$PY" -m vh_gopt.dataset.extract_features "$@" ;;
  train)    "$PY" -m vh_gopt.training.gopt_train "$@" ;;
  *)
    grep '^#   ' "$0" | sed 's/^# *//'
    exit 1 ;;
esac
