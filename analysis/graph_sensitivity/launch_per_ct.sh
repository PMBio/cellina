#!/usr/bin/env bash
# Per-cell-type neighbor-graph sensitivity sweep (within-domain).
#
# Runs run_sensitivity.py for each (cell_type, k, seed), pinned to a GPU via
# CUDA_VISIBLE_DEVICES, at most JOBS_PER_GPU procs per GPU. Same plain-bash slot
# scheduler as launch_sweep.sh, wrapped in a cell-type loop. Idempotent: skips
# any (ct, k, seed) whose result JSON already exists (so pre-seeded Myeloid
# within-domain results are reused, not recomputed).
#
# Results  -> $OUTROOT/<ct>/k{k}_seed{seed}.json
# Ckpts    -> $CKPTROOT/<ct>/k{k}_seed{seed}/
#
# Config via environment variables (all optional):
#   CTS           cell types             (default: 5 viable types)
#   KS            k grid                 (default: "5 10 100 200 1000 2000 10000")
#   SEEDS         seeds                  (default: "0 1 2")
#   GPUS          GPU ids                (default: "0 1")
#   JOBS_PER_GPU  concurrent procs/GPU   (default: 3)
#   OUTROOT       results root           (default: <script>/results_per_ct)
#   CKPTROOT      checkpoint root        (default: <script>/runs_per_ct)
#   LOGDIR        per-run stdout/stderr  (default: <script>/logs_per_ct)
#   PYTHON        interpreter            (default: cellina_edge env python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CTS="${CTS:-Fibroblast Endothelial Myeloid T_cell Epithelial}"
KS="${KS:-5 10 100 200 1000 2000 10000}"
SEEDS="${SEEDS:-0 1 2}"
GPUS="${GPUS:-0 1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
OUTROOT="${OUTROOT:-$SCRIPT_DIR/results_per_ct}"
CKPTROOT="${CKPTROOT:-$SCRIPT_DIR/runs_per_ct}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs_per_ct}"
PYTHON="${PYTHON:-/data/ddimitrov/software/miniforge3/envs/cellina_edge/bin/python}"

mkdir -p "$OUTROOT" "$CKPTROOT" "$LOGDIR"

# Flat list of GPU slots (e.g. "0 0 0 1 1 1"); job i -> slots[i % nslots].
SLOTS=()
for g in $GPUS; do
  for _ in $(seq 1 "$JOBS_PER_GPU"); do SLOTS+=("$g"); done
done
NSLOTS=${#SLOTS[@]}

echo "== per-cell-type graph-sensitivity sweep (within-domain) =="
echo "  CTS          : $CTS"
echo "  KS           : $KS"
echo "  SEEDS        : $SEEDS"
echo "  GPUS         : $GPUS  (x${JOBS_PER_GPU} jobs each -> $NSLOTS concurrent)"
echo "  OUTROOT      : $OUTROOT"
echo "  CKPTROOT     : $CKPTROOT"
echo "  LOGDIR       : $LOGDIR"
echo "  PYTHON       : $PYTHON"
echo

declare -a PIDS=()
for ((s=0; s<NSLOTS; s++)); do PIDS[$s]=""; done

FREE_SLOT=-1
wait_for_slot() {
  while true; do
    for ((s=0; s<NSLOTS; s++)); do
      local pid="${PIDS[$s]}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        FREE_SLOT=$s
        return 0
      fi
    done
    sleep 5
  done
}

njobs=0; nskip=0
for ct in $CTS; do
  outdir="$OUTROOT/$ct"
  ckpt_root="$CKPTROOT/$ct"
  mkdir -p "$outdir" "$ckpt_root"
  for k in $KS; do
    for seed in $SEEDS; do
      tag="k${k}_seed${seed}"
      if [[ -f "$outdir/$tag.json" ]]; then
        echo "[skip] $ct/$tag (result exists)"
        nskip=$((nskip + 1))
        continue
      fi
      wait_for_slot
      gpu="${SLOTS[$FREE_SLOT]}"
      echo "[launch] $ct/$tag on GPU $gpu (slot $FREE_SLOT)"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT_DIR/run_sensitivity.py" \
        --k "$k" --seed "$seed" --holdout-ct "$ct" \
        --outdir "$outdir" --ckpt-root "$ckpt_root" \
        >"$LOGDIR/${ct}_${tag}.log" 2>&1 &
      PIDS[$FREE_SLOT]=$!
      njobs=$((njobs + 1))
      sleep 2
    done
  done
done

echo
echo "Launched $njobs job(s), skipped $nskip; waiting for completion..."
wait
echo "== per-cell-type sweep complete =="
echo "Results in $OUTROOT ; logs in $LOGDIR"
