#!/usr/bin/env bash
# For HS/PF/BPNN — active state is SMEM, RFU lives in *.optstepN_v7 backups.
# This script:
#   1. Installs the named RFU optstep as active, runs, saves _rfu.log
#   2. Installs SMEM baseline (.bak_lds), runs, saves _smem.log
#   3. Leaves active = the chosen RFU (so future inspection shows RFU)
set -eo pipefail

DICE_ROOT="/data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/dice-test-gpu-rodinia/cuda"
DICE_TEST="$DICE_ROOT/dice_test"
SW20="$DICE_TEST/sw_20pe"
CFG_DICE="$DICE_TEST/cfg/gpgpusim_dice_rtx2060s.config"
CFG_ICNT="$DICE_TEST/cfg/config_turing_islip.icnt"

source /data2/jwang710/DICE-RFSMEM/DICE_ISCA_Eval/dice_gpgpu-sim/setup_environment > /dev/null

# bench → "rfu_suffix|smem_baseline_path"
# rfu_suffix: stem.meta.SUFFIX (e.g., optstep7_v7); pptx parallel
# smem_baseline: full path to baseline meta (pptx assumed at same stem with .meta→.pptx)
declare -A RFU_SUFFIX SMEM_META SMEM_PPTX
RFU_SUFFIX[hotspot]="optstep0_v7"
SMEM_META[hotspot]="$SW20/hotspot.1.sm_52.meta"
SMEM_PPTX[hotspot]="$SW20/hotspot.1.sm_52.pptx"
RFU_SUFFIX[pathfinder]="optstep7_v7"
SMEM_META[pathfinder]="$SW20/pathfinder.1.sm_52.meta"
SMEM_PPTX[pathfinder]="$SW20/pathfinder.1.sm_52.pptx"
RFU_SUFFIX[backprop]="optstep2_v7"
SMEM_META[backprop]="$SW20/backprop.1.sm_52.meta"
SMEM_PPTX[backprop]="$SW20/backprop.1.sm_52.pptx"

run_one() {
  local label="$1" bench="$2"
  local bench_dir="$DICE_ROOT/$bench"
  local report="$bench_dir/gpgpusim_dice_power_report.log"
  rm -f "$report"
  ( cd "$bench_dir" && . ./run > _${label}_run.log 2>&1 )
  if [ -f "$report" ]; then
    cp "$report" "$bench_dir/gpgpusim_dice_power_report_${label}.log"
    local cyc=$(grep -m1 "gpu_sim_cycle" "$bench_dir/_${label}_run.log" | awk '{print $3}')
    local dbb=$(grep -c "^DBB_ID" "$bench_dir/$bench.1.sm_52.meta")
    local match=$(grep -cE "(results match|GPU match|SUCCESS|0 mismatches|Maximum difference)" "$bench_dir/_${label}_run.log")
    echo "  [$bench/$label] DBBs=$dbb cycles=$cyc match_hits=$match"
  else
    echo "  [$bench/$label] FAILED — no report"
    tail -10 "$bench_dir/_${label}_run.log"
  fi
}

run_bench() {
  local bench="$1"
  local bench_dir="$DICE_ROOT/$bench"
  local stem="$bench.1.sm_52"
  local active_meta="$bench_dir/$stem.meta"
  local active_pptx="$bench_dir/$stem.pptx"
  local rfu_suffix="${RFU_SUFFIX[$bench]}"
  local rfu_meta="$bench_dir/$stem.meta.$rfu_suffix"
  local rfu_pptx="$bench_dir/$stem.pptx.$rfu_suffix"
  local smem_meta="${SMEM_META[$bench]}"
  local smem_pptx="${SMEM_PPTX[$bench]}"

  echo "=== [$bench] RFU=$rfu_suffix ==="
  for f in "$rfu_meta" "$rfu_pptx" "$smem_meta" "$smem_pptx"; do
    [ -f "$f" ] || { echo "  MISSING $f — skip"; return 1; }
  done

  # Ensure config in place (replicates copy_config)
  cp "$CFG_DICE" "$bench_dir/gpgpusim.config"
  cp "$CFG_ICNT" "$bench_dir/config_turing_islip.icnt"

  # Install RFU + run
  cp "$rfu_meta" "$active_meta"
  cp "$rfu_pptx" "$active_pptx"
  run_one "rfu" "$bench"

  # Install SMEM + run
  cp "$smem_meta" "$active_meta"
  cp "$smem_pptx" "$active_pptx"
  run_one "smem" "$bench"

  # Restore RFU as the durable active state
  cp "$rfu_meta" "$active_meta"
  cp "$rfu_pptx" "$active_pptx"
  echo "  [$bench] active state left = RFU ($rfu_suffix)"
}

for b in "$@"; do
  run_bench "$b" || echo "  FAILED $b"
done
