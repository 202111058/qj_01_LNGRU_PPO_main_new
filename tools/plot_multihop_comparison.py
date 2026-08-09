"""Create the three four-panel comparison figures with Multihop-RMAPPO.

Existing algorithms are loaded from the JSON embedded in the latest plot
scripts, preserving the manually curated values and scaling conventions in
those files.  The new algorithm is loaded from the exported text files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


METRICS = ("completion_rate", "system_time", "uav_energy", "usv_energy")
METRIC_PREFIX = {
    "completion_rate": "avg_completion_rate",
    "system_time": "avg_system_time",
    "uav_energy": "avg_uav_energy",
    "usv_energy": "avg_usv_energy",
}
Y_LABELS = {
    "completion_rate": "Task completion rate (%)",
    "system_time": "Task completion delay (s)",
    "uav_energy": "UAV energy consumption (J)",
    "usv_energy": "USV energy consumption (J)",
}
SUBLABELS = ("(a)", "(b)", "(c)", "(d)")
NEW_ALGO_KEY = "06_multihop_rmappo"
NEW_ALGO_LABEL = "MH-RMAPPO"

COLORS = {
    "01_lngru_ppo": "#E64B35",
    "02_ppo": "#00A087",
    "03_all_ppo": "#4DBBD5",
    "04_TD3": "#3C5488",
    "05_SAC": "#8491B4",
    "CORA": "#F39B7F",
    "LPPOUT": "#91D1C2",
    "PGTO": "#7E6148",
    "RD": "#B09C85",
    NEW_ALGO_KEY: "#8B1E6D",
}
LABELS = {
    "RD": "RD",
    "02_ppo": "PPO-COPG",
    "04_TD3": "TD3-COPG",
    "05_SAC": "SAC-COPG",
    "LPPOUT": "LPPOUT",
    "CORA": "CORA",
    "PGTO": "PGTO",
    "03_all_ppo": "PPO",
    "01_lngru_ppo": "LPPO-COPG",
    NEW_ALGO_KEY: NEW_ALGO_LABEL,
}
MARKERS = {
    "01_lngru_ppo": "o",
    "02_ppo": "s",
    "03_all_ppo": "^",
    "04_TD3": "D",
    "05_SAC": "v",
    "CORA": "p",
    "LPPOUT": "*",
    "PGTO": "h",
    "RD": "X",
    NEW_ALGO_KEY: "P",
}
PLOT_ORDER = list(LABELS)
BASELINE_SCALE_ALGOS = {"PGTO", "CORA", "RD"}

SWEEPS = {
    "task_size": {
        "template": Path("task_size/plot_4/plot_5.py"),
        "values": [0.4, 0.8, 1.2, 1.6, 2.0],
        "leaf": lambda value: f"task_size/task_size_{value:.1f}MB",
        "xlabel": "Average task size (Mb)",
        "stem": "task_size_final_with_multihop",
    },
    "uav_res": {
        "template": Path("uav_res/plot_4/plot_5.py"),
        "values": [5, 15, 25, 35, 45],
        "leaf": lambda value: f"uav_res/uav_res_{int(value)}",
        "xlabel": "Computing resources of UAVs (GHz)",
        "stem": "uav_res_final_with_multihop",
    },
    "usv_num": {
        "template": Path("USV_num/plot_4/plot_4.py"),
        "values": [10, 20, 30, 40, 50],
        "leaf": lambda value: f"usv_num/usv_num_{int(value)}",
        "xlabel": "The number of USVs",
        "stem": "usv_num_final_with_multihop",
    },
}


def load_embedded_json(template_path: Path) -> dict:
    source = template_path.read_text(encoding="utf-8")
    match = re.search(
        r'json_data_str\s*=\s*"""\s*(\{.*?\})\s*"""',
        source,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"cannot find json_data_str in {template_path}")
    return json.loads(match.group(1))


def mean_numeric_lines(path: Path) -> float | None:
    number_re = re.compile(
        r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
    )
    values: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if number_re.match(text):
            value = float(text)
            if math.isfinite(value):
                values.append(value)
    return sum(values) / len(values) if values else None


def load_new_metric(leaf: Path, metric: str) -> float | None:
    prefix = METRIC_PREFIX[metric]
    candidates = sorted(leaf.glob(f"{prefix}_*.txt"))
    if not candidates:
        return None
    values = [
        value
        for value in (mean_numeric_lines(path) for path in candidates)
        if value is not None
    ]
    return sum(values) / len(values) if values else None


def normalize_existing(
    sweep: str,
    x_value: float,
    metric: str,
    algorithm: str,
    value: float,
) -> float:
    if algorithm not in BASELINE_SCALE_ALGOS:
        return value
    if metric == "uav_energy":
        return value / 2.0
    if metric == "usv_energy":
        divisor = x_value if sweep == "usv_num" else 20.0
        return value / divisor
    return value


def normalize_new(
    sweep: str,
    x_value: float,
    metric: str,
    value: float,
) -> float:
    # train/system_time is the sum over all USVs.  The comparison figures use
    # per-USV task completion delay.
    if metric == "system_time":
        divisor = x_value if sweep == "usv_num" else 20.0
        return value / divisor
    return value


def build_plot_data(
    comparison_root: Path,
    sweep: str,
    algorithm_dir: str,
) -> Dict[str, Dict[float, Dict[str, float | None]]]:
    config = SWEEPS[sweep]
    raw = load_embedded_json(comparison_root / config["template"])
    result: Dict[str, Dict[float, Dict[str, float | None]]] = {}
    new_root = comparison_root / algorithm_dir
    for metric in METRICS:
        result[metric] = {}
        for x_value in config["values"]:
            raw_algorithms = raw[metric].get(str(x_value))
            if raw_algorithms is None and float(x_value).is_integer():
                raw_algorithms = raw[metric].get(str(int(x_value)))
            if raw_algorithms is None:
                raise KeyError(f"{sweep}/{metric}/{x_value} missing in template JSON")
            algorithms = {
                algorithm: normalize_existing(
                    sweep, float(x_value), metric, algorithm, float(value)
                )
                for algorithm, value in raw_algorithms.items()
            }
            leaf = new_root / config["leaf"](x_value)
            new_value = load_new_metric(leaf, metric) if leaf.is_dir() else None
            algorithms[NEW_ALGO_KEY] = (
                normalize_new(sweep, float(x_value), metric, new_value)
                if new_value is not None
                else None
            )
            result[metric][float(x_value)] = algorithms
    return result


def plot_sweep(
    comparison_root: Path,
    sweep: str,
    data: Mapping[str, Mapping[float, Mapping[str, float | None]]],
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = SWEEPS[sweep]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(1, 4, figsize=(20, 3.5))
    fig.subplots_adjust(wspace=0.24, top=0.79, bottom=0.25, left=0.05, right=0.99)
    x_values = [float(value) for value in config["values"]]

    for index, metric in enumerate(METRICS):
        axis = axes[index]
        for algorithm in PLOT_ORDER:
            y_values = [data[metric][x].get(algorithm) for x in x_values]
            if all(value is None for value in y_values):
                continue
            axis.plot(
                x_values,
                [math.nan if value is None else value for value in y_values],
                label=LABELS[algorithm],
                color=COLORS[algorithm],
                marker=MARKERS[algorithm],
                linestyle="-" if algorithm == NEW_ALGO_KEY else "--",
                linewidth=2.4 if algorithm == NEW_ALGO_KEY else 1.5,
                markersize=8 if algorithm == NEW_ALGO_KEY else 7,
                markeredgewidth=0,
                zorder=12 if algorithm == NEW_ALGO_KEY else 3,
            )
        axis.set_ylabel(Y_LABELS[metric], fontsize=14)
        axis.set_xlabel(config["xlabel"], fontsize=14)
        axis.set_xticks(x_values)
        axis.tick_params(axis="both", labelsize=12)
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.text(
            0.5,
            -0.22,
            SUBLABELS[index],
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=16,
        )
        if metric == "completion_rate":
            axis.set_ylim(0, 100)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = [LABELS[key] for key in PLOT_ORDER if LABELS[key] in by_label]
    fig.legend(
        [by_label[label] for label in ordered_labels],
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=10,
        frameon=True,
        fontsize=13,
        columnspacing=1.35,
        handlelength=2.0,
    )

    output_dir = (comparison_root / config["template"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = config["stem"]
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    json_path = output_dir / f"{stem}.json"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return [png_path, pdf_path, json_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--algorithm-dir", default=NEW_ALGO_KEY)
    parser.add_argument(
        "--dependency-path",
        type=Path,
        help="Optional directory containing the locally installed matplotlib",
    )
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=sorted(SWEEPS),
        default=sorted(SWEEPS),
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Refresh the explicit plot JSON without importing Matplotlib",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dependency_path:
        sys.path.insert(0, str(args.dependency_path.resolve()))
    comparison_root = args.comparison_root.resolve()
    for sweep in args.sweeps:
        data = build_plot_data(comparison_root, sweep, args.algorithm_dir)
        missing = [
            x
            for x, algorithms in data["completion_rate"].items()
            if algorithms.get(NEW_ALGO_KEY) is None
        ]
        if missing:
            print(f"MISSING {sweep} new-algorithm points: {missing}")
        if args.data_only:
            config = SWEEPS[sweep]
            output_dir = (comparison_root / config["template"]).parent
            json_path = output_dir / f"{config['stem']}.json"
            json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"WROTE {json_path}")
        else:
            for path in plot_sweep(comparison_root, sweep, data):
                print(f"WROTE {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
