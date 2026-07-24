#!/usr/bin/env bash
# Per-cell-type node-perturbation FRACTION sweep (within-domain, inference only).
#
# Runs run_node_fraction.py for each (cell_type, fraction, seed), pinned to a GPU
# via CUDA_VISIBLE_DEVICES, at most JOBS_PER_GPU procs per GPU. Same plain-bash
# slot scheduler as the graph_sensitivity launchers, wrapped in a cell-type loop.
# Idempotent: skips any (ct, fraction, seed) whose result JSON already exists.
#
# For each cell type it REUSES the k=200 within-domain checkpoint that was trained
# with that cell type held out (graph_sensitivity per-ct sweep). No retraining:
#   Myeloid -> graph_sensitivity/runs_within_domain/k200_seed{seed}
#   others  -> graph_sensitivity/runs_per_ct/<ct>/k200_seed{seed}
# The held-out cell type drives the tutorial logFC definition (excluded from the
# global shift; its own row overridden by that global shift) -- see
# run_node_fraction.py / docs/tutorial.ipynb 4.2.
#
# Results -> $OUTROOT/<ct>/frac{fraction}_seed{seed}.json
#
# Config via environment variables (all optional):
#   CTS           cell types             (default: 5 viable types)
#   FRACTIONS     perturb_fraction grid  (default: "0.05 0.1 0.25 0.5 0.75 1.0")
#   SEEDS         seeds                  (default: "0 1 2")
#   GPUS          GPU ids                (default: "0 1")
#   JOBS_PER_GPU  concurrent procs/GPU   (default: 3)
#   OUTROOT       results root           (default: <script>/results_per_ct)
#   LOGDIR        per-run stdout/stderr  (default: <script>/logs_per_ct)
#   GS_DIR        graph_sensitivity dir  (default: <script>/../graph_sensitivity)
#   PYTHON        interpreter            (default: cellina_edge env python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CTS="${CTS:-Fibroblast Endothelial Myeloid T_cell Epithelial}"
FRACTIONS="${FRACTIONS:-0.05 0.1 0.25 0.5 0.75 1.0}"
SEEDS="${SEEDS:-0 1 2}"
GPUS="${GPUS:-0 1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
OUTROOT="${OUTROOT:-$SCRIPT_DIR/results_per_ct}"
LOGDIR="${LOGDIR:-$SCRIPT_DIR/logs_per_ct}"
GS_DIR="${GS_DIR:-$(cd "$SCRIPT_DIR/../graph_sensitivity" && pwd)}"
PYTHON="${PYTHON:-/data/ddimitrov/software/miniforge3/envs/cellina_edge/bin/python}"

mkdir -p "$OUTROOT" "$LOGDIR"

# Per-ct checkpoint root (Myeloid lives outside runs_per_ct; see header).
ckpt_root_for() {
  local ct="$1"
  if [[ "$ct" == "Myeloid" ]]; then
    echo "$GS_DIR/runs_within_domain"
  else
    echo "$GS_DIR/runs_per_ct/$ct"
  fi
}

# Flat list of GPU slots (e.g. "0 0 0 1 1 1"); job i -> slots[i % nslots].
SLOTS=()
for g in $GPUS; do
  for _ in $(seq 1 "$JOBS_PER_GPU"); do SLOTS+=("$g"); done
done
NSLOTS=${#SLOTS[@]}

echo "== per-cell-type node-fraction sweep (within-domain, inference only) =="
echo "  CTS          : $CTS"
echo "  FRACTIONS    : $FRACTIONS"
echo "  SEEDS        : $SEEDS"
echo "  GPUS         : $GPUS  (x${JOBS_PER_GPU} jobs each -> $NSLOTS concurrent)"
echo "  OUTROOT      : $OUTROOT"
echo "  GS_DIR       : $GS_DIR"
echo "  LOGDIR       : $LOGDIR"
echo "  PYTHON       : $PYTHON"
echo

# Fail fast if any reused checkpoint is missing.
for ct in $CTS; do
  cr="$(ckpt_root_for "$ct")"
  for seed in $SEEDS; do
    if [[ ! -d "$cr/k200_seed${seed}" ]]; then
      echo "ERROR: missing checkpoint $cr/k200_seed${seed} for $ct" >&2
      exit 1
    fi
  done
done

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
  ckpt_root="$(ckpt_root_for "$ct")"
  mkdir -p "$outdir"
  for frac in $FRACTIONS; do
    for seed in $SEEDS; do
      tag="frac${frac}_seed${seed}"
      if [[ -f "$outdir/$tag.json" ]]; then
        echo "[skip] $ct/$tag (result exists)"
        nskip=$((nskip + 1))
        continue
      fi
      wait_for_slot
      gpu="${SLOTS[$FREE_SLOT]}"
      echo "[launch] $ct/$tag on GPU $gpu (slot $FREE_SLOT)"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT_DIR/run_node_fraction.py" \
        --fraction "$frac" --seed "$seed" --holdout-ct "$ct" \
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
echo "== per-cell-type node-fraction sweep complete =="
echo "Results in $OUTROOT ; logs in $LOGDIR"
echo "Next: $PYTHON $SCRIPT_DIR/plot_per_ct.py --results $OUTROOT"
