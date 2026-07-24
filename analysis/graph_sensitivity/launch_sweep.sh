#!/usr/bin/env bash
# Launch the neighbor-graph sensitivity sweep across local GPUs.
#
# Runs one `run_sensitivity.py` process per (k, seed) pair, pinned to a GPU via
# CUDA_VISIBLE_DEVICES, with at most JOBS_PER_GPU processes per GPU running at
# once. Plain-bash scheduler: fills GPU slots, waits for a free slot, repeats.
#
# Within-domain graph (library_key='typ' per-domain kNN): no cross-domain edges.
# Writes to within-domain dirs by default so the old mixed-domain runs/results
# are preserved (you can go back).
#
# Config via environment variables (all optional):
#   KS            k grid                 (default: "5 10 100 200 1000 2000 10000")
#   SEEDS         seeds                  (default: "0 1 2")
#   GPUS          GPU ids to use         (default: "0 1")
#   JOBS_PER_GPU  concurrent procs/GPU   (default: 3)
#   OUTDIR        results dir            (default: <script>/results_within_domain)
#   CKPT_ROOT     checkpoint root        (default: <script>/runs_within_domain)
#   LOGDIR        per-run stdout/stderr  (default: <script>/logs_within_domain)
#   PYTHON        interpreter            (default: cellina_edge env python)
#
# Example:
#   bash launch_sweep.sh                      # full 21-run sweep, 3 jobs/GPU
#   JOBS_PER_GPU=2 GPUS="0" bash launch_sweep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

KS="${KS:-5 10 100 200 1000 2000 10000}"
SEEDS="${SEEDS:-0 1 2}"
GPUS="${GPUS:-0 1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/results_within_domain}"
CKPT_ROOT="${CKPT_ROOT:-$SCRIPT_DIR/runs_within_domain}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs_within_domain}"
PYTHON="${PYTHON:-/data/ddimitrov/software/miniforge3/envs/cellina_edge/bin/python}"

mkdir -p "$OUTDIR" "$CKPT_ROOT" "$LOGDIR"

# Build the flat list of GPU slots, e.g. GPUS="0 1", JOBS_PER_GPU=3 ->
# slots = (0 0 0 1 1 1). Job i is assigned to slots[i % nslots].
SLOTS=()
for g in $GPUS; do
  for _ in $(seq 1 "$JOBS_PER_GPU"); do SLOTS+=("$g"); done
done
NSLOTS=${#SLOTS[@]}

echo "== graph-sensitivity sweep (within-domain) =="
echo "  KS           : $KS"
echo "  SEEDS        : $SEEDS"
echo "  GPUS         : $GPUS  (x${JOBS_PER_GPU} jobs each -> $NSLOTS concurrent)"
echo "  OUTDIR       : $OUTDIR"
echo "  CKPT_ROOT    : $CKPT_ROOT"
echo "  LOGDIR       : $LOGDIR"
echo "  PYTHON       : $PYTHON"
echo

declare -a PIDS=()   # pid per slot (empty string = free)
for ((s=0; s<NSLOTS; s++)); do PIDS[$s]=""; done

# Block until at least one slot is free; return its index via FREE_SLOT.
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

njobs=0
for k in $KS; do
  for seed in $SEEDS; do
    tag="k${k}_seed${seed}"
    # Skip already-completed runs (idempotent re-launch).
    if [[ -f "$OUTDIR/$tag.json" ]]; then
      echo "[skip] $tag (result exists)"
      continue
    fi
    wait_for_slot
    gpu="${SLOTS[$FREE_SLOT]}"
    echo "[launch] $tag on GPU $gpu (slot $FREE_SLOT)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT_DIR/run_sensitivity.py" \
      --k "$k" --seed "$seed" --outdir "$OUTDIR" --ckpt-root "$CKPT_ROOT" \
      >"$LOGDIR/$tag.log" 2>&1 &
    PIDS[$FREE_SLOT]=$!
    njobs=$((njobs + 1))
    sleep 2   # stagger starts so concurrent HVG/graph builds don't collide on RAM
  done
done

echo
echo "Launched $njobs job(s); waiting for completion..."
wait
echo "== sweep complete =="
echo "Results in $OUTDIR ; logs in $LOGDIR"
echo "Next: $PYTHON $SCRIPT_DIR/plot_sensitivity.py --results $OUTDIR"
