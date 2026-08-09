#!/bin/bash
# Sweep over task sizes (MB): label(avg), min-max
#   0.4, 0.2-0.6
#   0.8, 0.5-1.0
#   1.2, 0.8-1.6
#   1.6, 1.1-2.1
#   2.0, 1.4-2.6
# Usage: bash scripts/sweep_task_size.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

NUM_ENV_STEPS=300000
SEED=1
SESSION="sweep_tasksize"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "task04"

tmux send-keys -t "${SESSION}:task04" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --task_size_max 0.6 --task_size_min 0.2 --task_size_label 0.4 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "task08"
tmux send-keys -t "${SESSION}:task08" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --task_size_max 1.0 --task_size_min 0.5 --task_size_label 0.8 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "task12"
tmux send-keys -t "${SESSION}:task12" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --task_size_max 1.6 --task_size_min 0.8 --task_size_label 1.2 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "task16"
tmux send-keys -t "${SESSION}:task16" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --task_size_max 2.1 --task_size_min 1.1 --task_size_label 1.6 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "task20"
tmux send-keys -t "${SESSION}:task20" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --task_size_max 2.6 --task_size_min 1.4 --task_size_label 2.0 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

echo "All task-size sweep experiments launched in tmux session '${SESSION}'."
echo "Use 'tmux attach -t ${SESSION}' to monitor."
