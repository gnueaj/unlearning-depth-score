#!/bin/bash
cd /home/jaeung/activation-patching-unlearning

echo "$(date) Waiting for UDS faithfulness (SDPA) to complete..."

while true; do
  g0=$(python3 -c "import json; print(len(json.load(open('runs/meta_eval/table2_faithfulness_uds_sdpa_gpu0/results.json'))))" 2>/dev/null || echo 0)
  g1=$(python3 -c "import json; print(len(json.load(open('runs/meta_eval/table2_faithfulness_uds_sdpa_gpu1/results.json'))))" 2>/dev/null || echo 0)
  echo "$(date '+%H:%M:%S') UDS: GPU0=$g0/30 GPU1=$g1/30"
  
  if [ "$g0" = "30" ] && [ "$g1" = "30" ]; then
    echo "$(date) UDS complete!"
    break
  fi
  sleep 60
done

sleep 5

echo "$(date) Resuming EP10..."
python3 scripts/run_ep10_experiments.py --gpu 0 --start 0 --end 38 > logs/ep10_metrics_gpu0.log 2>&1 &
python3 scripts/run_ep10_experiments.py --gpu 1 --start 38 --end 75 > logs/ep10_metrics_gpu1.log 2>&1 &
wait
echo "$(date) EP10 complete!"
