#!/usr/bin/env bash
# Launch the node-perturbation fraction sweep across local GPUs.
#
# One `run_node_fraction.py` process per (fraction, seed), pinned to a GPU via
# CUDA_VISIBLE_DEVICES, at most JOBS_PER_GPU processes per GPU. Inference only
# (reuses the pre-trained k=200 checkpoints) -- each run is fast.
#
# Config via env vars (all optional):
#   FRACTIONS     perturb_fraction grid  (default: "0.05 0.1 0.25 0.5 1.0")
#   SEEDS         model seeds            (default: "0 1 2")
#   GPUS          GPU ids                (default: "0 1")
#   JOBS_PER_GPU  concurrent procs/GPU   (default: 3)
#   OUTDIR        results dir            (default: <script>/results)
#   LOGDIR        per-run logs           (default: <script>/logs)
#   PYTHON        interpreter            (default: cellina_edge env python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FRACTIONS="${FRACTIONS:-0.05 0.1 0.25 0.5 1.0}"
SEEDS="${SEEDS:-0 1 2}"
GPUS="${GPUS:-0 1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
OUTDIR="${OUTDIR:-$SCRIPT_DIR/results}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs}"
PYTHON="${PYTHON:-/data/ddimitrov/software/miniforge3/envs/cellina_edge/bin/python}"

mkdir -p "$OUTDIR" "$LOGDIR"

SLOTS=()
for g in $GPUS; do
  for _ in $(seq 1 "$JOBS_PER_GPU"); do SLOTS+=("$g"); done
done
NSLOTS=${#SLOTS[@]}

echo "== node-fraction sweep =="
echo "  FRACTIONS    : $FRACTIONS"
echo "  SEEDS        : $SEEDS"
echo "  GPUS         : $GPUS  (x${JOBS_PER_GPU} -> $NSLOTS concurrent)"
echo "  OUTDIR       : $OUTDIR"
echo

declare -a PIDS=()
for ((s=0; s<NSLOTS; s++)); do PIDS[$s]=""; done

FREE_SLOT=-1
wait_for_slot() {
  while true; do
    for ((s=0; s<NSLOTS; s++)); do
      local pid="${PIDS[$s]}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        FREE_SLOT=$s; return 0
      fi
    done
    sleep 3
  done
}

njobs=0
for frac in $FRACTIONS; do
  for seed in $SEEDS; do
    tag="frac${frac}_seed${seed}"
    if [[ -f "$OUTDIR/$tag.json" ]]; then
      echo "[skip] $tag (result exists)"; continue
    fi
    wait_for_slot
    gpu="${SLOTS[$FREE_SLOT]}"
    echo "[launch] $tag on GPU $gpu (slot $FREE_SLOT)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT_DIR/run_node_fraction.py" \
      --seed "$seed" --fraction "$frac" --outdir "$OUTDIR" \
      >"$LOGDIR/$tag.log" 2>&1 &
    PIDS[$FREE_SLOT]=$!
    njobs=$((njobs + 1))
    sleep 2
  done
done

echo; echo "Launched $njobs job(s); waiting..."
wait
echo "== sweep complete =="
echo "Next: $PYTHON $SCRIPT_DIR/plot_node_fraction.py --results $OUTDIR"
