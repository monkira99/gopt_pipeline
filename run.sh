#!/usr/bin/env bash
# Dispatcher for vh-gopt pipeline (UV-native / virtualenv).
#   ./run.sh install                    # cai dat package editable (uv pip install -e .)
#   ./run.sh snapshot [flags...]        # (may co du lieu thoi) tao + push corpus snapshot len HF
#   ./run.sh fetch    [flags...]        # (server build) tai corpus snapshot tu HF
#   ./run.sh pack     [flags...]        # corpus -> 4 npz (mac dinh trich GOP 80-d; --skip-gop de nhan-only)
#   ./run.sh verify   [flags...]        # cong chat luong truoc khi push
#   ./run.sh push     [flags...]        # npz -> datasets.DatasetDict -> HF Hub (--dry-run de thu conversion)
#   ./run.sh build    [flags...]        # npz nhan + audio -> GOLD Arrow (audio+labels, load_dataset duoc); --repo de push
#   ./run.sh extract  [flags...]        # (server GPU) trich day du Feature: KoelLabs 80-d + Prosody 8-d + WavLM 1024-d -> 4 .npz
#   ./run.sh train    [flags...]        # train GOPT/HIA tren npz trung gian
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# Chon runner: Uu tien 'uv run', neu khong co uv thi dung .venv hoac python3
if [ -n "${PY:-}" ]; then
  RUNNER=("$PY")
elif command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run --project "$ROOT" python)
elif [ -x "$ROOT/.venv/bin/python" ]; then
  RUNNER=("$ROOT/.venv/bin/python")
elif [ -x "$ROOT/../.venv/bin/python" ]; then
  RUNNER=("$ROOT/../.venv/bin/python")
else
  RUNNER=(python3)
fi

cmd="${1:-}"; shift || true
case "$cmd" in
  install)
    if command -v uv >/dev/null 2>&1; then
      uv pip install -e "$ROOT"
    else
      "${RUNNER[@]}" -m pip install -e "$ROOT"
    fi
    ;;
  snapshot) "${RUNNER[@]}" -m vh_gopt.dataset.snapshot_corpus "$@" ;;
  fetch)    "${RUNNER[@]}" -m vh_gopt.dataset.fetch_corpus "$@" ;;
  pack)     "${RUNNER[@]}" -m vh_gopt.dataset.pack_stage2 "$@" ;;
  verify)   "${RUNNER[@]}" -m vh_gopt.dataset.verify_dataset "$@" ;;
  push)     "${RUNNER[@]}" -m vh_gopt.dataset.push_dataset "$@" ;;
  build)    "${RUNNER[@]}" -m vh_gopt.dataset.build_gold_arrow "$@" ;;
  extract)  "${RUNNER[@]}" -m vh_gopt.dataset.extract_features "$@" ;;
  train)    "${RUNNER[@]}" -m vh_gopt.training.gopt_train "$@" ;;
  *)
    grep '^#   ' "$0" | sed 's/^# *//'
    exit 1 ;;
esac
