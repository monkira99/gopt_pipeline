#!/usr/bin/env bash
# Grid đa-seed cho model hợp nhất: baseline (use_occ=false) vs +occ, seeds 0/1/2.
# Mỗi run ghi ckpt/<tag>/metrics.json. Chạy: bash run_unified_grid.sh  (nền tốt nhất: nohup ... &)
# PY overridable: local MPS -> venv; RunPod -> python. KHÔNG set -e (1 seed lỗi không giết cả grid).
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
CFG=configs/stage2/unified.yaml
run(){ # tag seed extra...
  local tag=$1 seed=$2; shift 2
  echo "===== RUN $tag (seed=$seed) ====="
  $PY -m vh_gopt.training.unified_train --config $CFG --seed $seed \
      --out ckpt/$tag --no-wandb "$@" 2>&1 | grep -E "^(VAL|TEST|det counts|model=)" || true
}
for s in 0 1 2; do run base_s$s $s; done
for s in 0 1 2; do run occ_s$s  $s --use-occ; done
echo "===== GRID DONE ====="
$PY collect_unified.py
