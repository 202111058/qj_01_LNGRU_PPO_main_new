#!/bin/bash
# Sweep over UAV resource (GHz): 5, 15, 25, 35, 45
# Usage: bash scripts/sweep_uav_res.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

NUM_ENV_STEPS=300000
SEED=1
SESSION="sweep_uavres"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "res5"

tmux send-keys -t "${SESSION}:res5" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --uav_resource 5 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "res15"
tmux send-keys -t "${SESSION}:res15" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --uav_resource 15 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "res25"
tmux send-keys -t "${SESSION}:res25" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --uav_resource 25 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "res35"
tmux send-keys -t "${SESSION}:res35" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --uav_resource 35 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "res45"
tmux send-keys -t "${SESSION}:res45" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --uav_resource 45 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

echo "All UAV-resource sweep experiments launched in tmux session '${SESSION}'."
echo "Use 'tmux attach -t ${SESSION}' to monitor."
