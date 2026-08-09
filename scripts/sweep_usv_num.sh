#!/bin/bash
# Sweep over USV numbers: 10, 20, 30, 40, 50
# Usage: bash scripts/sweep_usv_num.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

NUM_ENV_STEPS=300000
SEED=1
SESSION="sweep_usv"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "usv10"

tmux send-keys -t "${SESSION}:usv10" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --num_usv 10 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "usv20"
tmux send-keys -t "${SESSION}:usv20" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --num_usv 20 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "usv30"
tmux send-keys -t "${SESSION}:usv30" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --num_usv 30 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "usv40"
tmux send-keys -t "${SESSION}:usv40" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --num_usv 40 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

tmux new-window -t "$SESSION" -n "usv50"
tmux send-keys -t "${SESSION}:usv50" \
    "cd ${SCRIPT_DIR} && python train/train_multihop_sweep.py --num_usv 50 --num_env_steps ${NUM_ENV_STEPS} --seed ${SEED} --algorithm_name rmappo --use_recurrent_policy" Enter

echo "All USV sweep experiments launched in tmux session '${SESSION}'."
echo "Use 'tmux attach -t ${SESSION}' to monitor."
