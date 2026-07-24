#!/usr/bin/env bash
# One-shot status snapshot for the node-fraction experiment.
cd /data/ddimitrov/repos/cellina/analysis/node_fraction
echo "===== node-fraction status @ $(date '+%H:%M:%S') ====="
echo "-- smoke --"
if [ -f results_smoke/frac0.5_seed0.json ]; then
  echo "smoke: DONE"; else echo "smoke: pending/running"; fi
echo "-- sweep results ($(ls -1 results/*.json 2>/dev/null | wc -l)/15) --"
ls -1 results/*.json 2>/dev/null | sed 's#.*/##;s#.json##' | tr '\n' ' '; echo
echo "-- running workers --"
pgrep -af "run_node_fraction.py" | grep -v pgrep | sed 's#.*run_node_fraction.py##' || echo "  none"
echo "-- errors in logs --"
grep -liE "Traceback|Error" logs/*.log 2>/dev/null | sed 's#.*/##' || echo "  none"
echo "-- GPU --"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
