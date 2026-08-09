"""Export a 1000-point event and update plot_reward_delay_3.py to use it."""

from __future__ import annotations

import argparse
from pathlib import Path

from export_multihop_comparison import read_scalars


REWARD_TAG = "train/episode_reward"
DELAY_TAG = "train/system_time"
REWARD_RELATIVE = Path("icc_reward/07_multi_hop_rewards_1000.txt")
DELAY_RELATIVE = Path("icc_avg_delay/07_multi_hop_avg_delay_1000.txt")


def write_values(path: Path, points: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{value:.6f}" for _, value in points) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def update_plot_script(plot_script: Path) -> None:
    source = plot_script.read_text(encoding="utf-8")
    replacements = {
        "'step_interval': 5,": "'step_interval': 1,",
        "'reward_file': 'icc_reward/07_multi_hop_rewards.txt',": (
            "'reward_file': 'icc_reward/07_multi_hop_rewards_1000.txt',"
        ),
        "'delay_file': 'icc_avg_delay/07_multi_hop_avg_delay.txt'": (
            "'delay_file': 'icc_avg_delay/07_multi_hop_avg_delay_1000.txt'"
        ),
        "reward_delay_combined_v3.pdf": "reward_delay_combined_v3_1000.pdf",
        "reward_delay_combined_v3.png": "reward_delay_combined_v3_1000.png",
    }
    for old, new in replacements.items():
        if old not in source and new not in source:
            raise ValueError(f"expected configuration text not found: {old}")
        source = source.replace(old, new)
    plot_script.write_text(source, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, type=Path)
    parser.add_argument("--reward-delay-root", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_path = args.event.resolve()
    output_root = args.reward_delay_root.resolve()
    scalars = read_scalars([event_path])
    reward_points = scalars.get(REWARD_TAG, [])
    delay_points = scalars.get(DELAY_TAG, [])

    if len(reward_points) != args.expected_count:
        raise SystemExit(
            f"{REWARD_TAG} has {len(reward_points)} points, "
            f"expected {args.expected_count}"
        )
    if len(delay_points) != args.expected_count:
        raise SystemExit(
            f"{DELAY_TAG} has {len(delay_points)} points, "
            f"expected {args.expected_count}"
        )
    reward_steps = [step for step, _ in reward_points]
    delay_steps = [step for step, _ in delay_points]
    if reward_steps != delay_steps:
        raise SystemExit("reward and delay steps do not match")

    reward_path = output_root / REWARD_RELATIVE
    delay_path = output_root / DELAY_RELATIVE
    plot_script = output_root / "plot_reward_delay_3.py"
    write_values(reward_path, reward_points)
    write_values(delay_path, delay_points)
    update_plot_script(plot_script)

    manifest = output_root / "multihop_1000_source.txt"
    manifest.write_text(
        "\n".join(
            [
                f"event={event_path}",
                f"point_count={len(reward_points)}",
                f"step_first={reward_steps[0]}",
                f"step_last={reward_steps[-1]}",
                f"reward_file={reward_path}",
                f"delay_file={delay_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"WROTE {len(reward_points)} reward points: "
        f"{reward_path} ({reward_steps[0]}..{reward_steps[-1]})"
    )
    print(
        f"WROTE {len(delay_points)} delay points: "
        f"{delay_path} ({delay_steps[0]}..{delay_steps[-1]})"
    )
    print(f"UPDATED {plot_script}")
    print(f"WROTE {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
