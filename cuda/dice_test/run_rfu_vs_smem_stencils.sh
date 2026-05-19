#!/usr/bin/env bash
# RFU vs SMEM-baseline DICEwattch sweep.
# Bypasses `make test_dice` because it includes a copy_meta step that
# regenerates .meta/.pptx from PTX, overwriting any swap we apply. Instead we
# run `./run` directly (as OPTIMIZATION_LOG.md prescribes), with the right
# gpgpusim.config + accelwattch xml already in place.
#
# Usage:  bash /tmp/run_rfu_vs_smem.sh <bench> [<bench> ...]
set -eo pipefail

DICE_ROOT="/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/dice-test-gpu-rodinia/cuda"
DICE_TEST="$DICE_ROOT/dice_test"
SW20="$DICE_TEST/sw_20pe"
CFG_DICE="$DICE_TEST/cfg/gpgpusim_dice_rtx2060s.config"
CFG_ICNT="$DICE_TEST/cfg/config_turing_islip.icnt"

declare -A BASELINE_META BASELINE_PPTX
BASELINE_META[stencil1d]="$SW20/stencil1d.1.sm_52.meta.lds_baseline"
BASELINE_PPTX[stencil1d]="$SW20/stencil1d.1.sm_52.pptx.lds_baseline"
BASELINE_META[stencil2d]="$DICE_ROOT/stencil2d/stencil2d.1.sm_52.meta.bak_lds"
BASELINE_PPTX[stencil2d]="$DICE_ROOT/stencil2d/stencil2d.1.sm_52.pptx.bak_lds"
BASELINE_META[stencil3d]="$DICE_ROOT/stencil3d/stencil3d.1.sm_52.meta.bak_lds"
BASELINE_PPTX[stencil3d]="$DICE_ROOT/stencil3d/stencil3d.1.sm_52.pptx.bak_lds"
BASELINE_META[hotspot]="$SW20/hotspot.1.sm_52.meta"
BASELINE_PPTX[hotspot]="$SW20/hotspot.1.sm_52.pptx"
BASELINE_META[pathfinder]="$SW20/pathfinder.1.sm_52.meta"
BASELINE_PPTX[pathfinder]="$SW20/pathfinder.1.sm_52.pptx"
BASELINE_META[backprop]="$SW20/backprop.1.sm_52.meta"
BASELINE_PPTX[backprop]="$SW20/backprop.1.sm_52.pptx"

source /data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/dice_gpgpu-sim/setup_environment > /dev/null

run_one() {
  local label="$1" bench="$2" report="$3"
  local bench_dir="$DICE_ROOT/$bench"
  rm -f "$report"
  ( cd "$bench_dir" && . ./run > _${label}_run.log 2>&1 )
  if [ -f "$report" ]; then
    cp "$report" "$bench_dir/gpgpusim_dice_power_report_${label}.log"
    local cyc=$(grep -m1 "gpu_sim_cycle" "$bench_dir/_${label}_run.log" | head -1 | awk '{print $3}')
    local dbb=$(grep -c "^DBB_ID" "$bench_dir/$bench.1.sm_52.meta")
    echo "  [$bench/$label] DBBs=$dbb cycles=$cyc"
  else
    echo "  [$bench/$label] FAILED — no report produced"
    tail -10 "$bench_dir/_${label}_run.log"
  fi
}

run_bench() {
  local bench="$1"
  local bench_dir="$DICE_ROOT/$bench"
  local stem="$bench.1.sm_52"
  local active_meta="$bench_dir/$stem.meta"
  local active_pptx="$bench_dir/$stem.pptx"
  local smem_meta="${BASELINE_META[$bench]}"
  local smem_pptx="${BASELINE_PPTX[$bench]}"
  local report="$bench_dir/gpgpusim_dice_power_report.log"
  local backup_meta="$bench_dir/$stem.meta.RFU_BACKUP"
  local backup_pptx="$bench_dir/$stem.pptx.RFU_BACKUP"

  echo "=== [$bench] ==="
  for f in "$smem_meta" "$smem_pptx" "$active_meta" "$active_pptx"; do
    [ -f "$f" ] || { echo "  MISSING $f — skipping $bench"; return 1; }
  done

  # Ensure gpgpusim.config + icnt are in place (replicates copy_config).
  cp "$CFG_DICE" "$bench_dir/gpgpusim.config"
  cp "$CFG_ICNT" "$bench_dir/config_turing_islip.icnt"

  # Snapshot current (RFU)
  cp "$active_meta" "$backup_meta"
  cp "$active_pptx" "$backup_pptx"

  # RFU run (current files)
  run_one "rfu" "$bench" "$report"

  # Swap to SMEM baseline
  cp "$smem_meta" "$active_meta"
  cp "$smem_pptx" "$active_pptx"
  run_one "smem" "$bench" "$report"

  # Restore RFU
  cp "$backup_meta" "$active_meta"
  cp "$backup_pptx" "$active_pptx"
  rm -f "$backup_meta" "$backup_pptx"
  echo "  [$bench] restored RFU state."
}

for b in "$@"; do
  run_bench "$b" || echo "  FAILED $b"
done
