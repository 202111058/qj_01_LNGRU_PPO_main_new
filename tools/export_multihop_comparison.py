"""Export MultihopUAVBS TensorBoard scalars into comparison-result folders.

The script intentionally uses only the Python standard library.  It reads the
simple scalar values stored in TensorBoard event files and writes the same
one-value-per-line text layout used by Journal_result_comparison.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple


EXPERIMENT_RE = re.compile(
    r"^usv(?P<usv>\d+)_task(?P<task_upper>\d+(?:\.\d+)?)M_"
    r"res(?P<uav_res>\d+)G_seed(?P<seed>\d+)$"
)

# The experiment folder records the upper bound.  The comparison x-axis uses
# the intended average-task-size label from scripts/sweep_task_size.sh.
TASK_UPPER_TO_AVERAGE = {
    0.6: 0.4,
    1.0: 0.8,
    1.6: 1.2,
    2.1: 1.6,
    2.6: 2.0,
}
TASK_AVERAGE_TO_RANGE = {
    0.4: (0.2, 0.6),
    0.8: (0.5, 1.0),
    1.2: (0.8, 1.6),
    1.6: (1.1, 2.1),
    2.0: (1.4, 2.6),
}

DEFAULT_USV = 20
DEFAULT_UAV_RES = 25
DEFAULT_TASK_AVERAGE = 0.8

TAG_TO_METRIC = {
    "train/completion_rate": "avg_completion_rate",
    "train/episode_reward": "avg_reward",
    "train/system_time": "avg_system_time",
    "train/total_energy": "avg_total_energy",
    "train/avg_uav_comp_energy": "avg_uav_comp_energy",
    "train/avg_uav_energy": "avg_uav_energy",
    "train/avg_uav_fly_energy": "avg_uav_fly_energy",
    "train/avg_uav_relay_energy": "avg_uav_relay_energy",
    "train/avg_usv_energy": "avg_usv_energy",
}


@dataclass(frozen=True)
class Experiment:
    path: Path
    usv: int
    task_upper: float
    task_average: float
    uav_res: int
    seed: int


def _read_varint(data: bytes, offset: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            raise ValueError("invalid protobuf varint")
    raise ValueError("truncated protobuf varint")


def _protobuf_fields(data: bytes) -> Iterator[Tuple[int, int, object]]:
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            if offset + 8 > len(data):
                raise ValueError("truncated protobuf fixed64")
            value = data[offset : offset + 8]
            offset += 8
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            if offset + length > len(data):
                raise ValueError("truncated protobuf bytes")
            value = data[offset : offset + length]
            offset += length
        elif wire_type == 5:
            if offset + 4 > len(data):
                raise ValueError("truncated protobuf fixed32")
            value = data[offset : offset + 4]
            offset += 4
        else:
            raise ValueError(f"unsupported protobuf wire type: {wire_type}")
        yield field_number, wire_type, value


def _summary_scalars(summary_data: bytes) -> Iterator[Tuple[str, float]]:
    for field_number, wire_type, raw_value in _protobuf_fields(summary_data):
        if field_number != 1 or wire_type != 2:
            continue
        tag = None
        scalar = None
        for value_field, value_wire, value_raw in _protobuf_fields(raw_value):
            if value_field == 1 and value_wire == 2:
                tag = value_raw.decode("utf-8", errors="replace")
            elif value_field == 2 and value_wire == 5:
                scalar = struct.unpack("<f", value_raw)[0]
        if tag is not None and scalar is not None:
            yield tag, scalar


def _event_scalars(event_data: bytes) -> Iterator[Tuple[int, str, float]]:
    step = 0
    summaries: List[bytes] = []
    for field_number, wire_type, raw_value in _protobuf_fields(event_data):
        if field_number == 2 and wire_type == 0:
            step = int(raw_value)
        elif field_number == 5 and wire_type == 2:
            summaries.append(raw_value)
    for summary in summaries:
        for tag, value in _summary_scalars(summary):
            yield step, tag, value


def iter_event_records(event_path: Path) -> Iterator[bytes]:
    with event_path.open("rb") as stream:
        while True:
            length_bytes = stream.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise ValueError(f"truncated TFRecord length in {event_path}")
            length = struct.unpack("<Q", length_bytes)[0]
            if len(stream.read(4)) != 4:
                raise ValueError(f"truncated TFRecord length CRC in {event_path}")
            payload = stream.read(length)
            if len(payload) != length:
                raise ValueError(f"truncated TFRecord payload in {event_path}")
            if len(stream.read(4)) != 4:
                raise ValueError(f"truncated TFRecord data CRC in {event_path}")
            yield payload


def read_scalars(event_paths: Sequence[Path]) -> Dict[str, List[Tuple[int, float]]]:
    # Later records for a repeated step win, matching TensorBoard's practical
    # behavior when logs resume or several event files exist in one run.
    values: Dict[str, Dict[int, float]] = defaultdict(dict)
    for event_path in sorted(event_paths, key=lambda path: path.stat().st_mtime):
        for record in iter_event_records(event_path):
            for step, tag, value in _event_scalars(record):
                if math.isfinite(value):
                    values[tag][step] = value
    return {
        tag: sorted(step_values.items())
        for tag, step_values in values.items()
    }


def find_latest_run_events(experiment_path: Path) -> Tuple[str, List[Path]]:
    runs: List[Tuple[int, Path]] = []
    for path in experiment_path.iterdir():
        match = re.fullmatch(r"run(\d+)", path.name) if path.is_dir() else None
        if match:
            runs.append((int(match.group(1)), path))
    for _, run_path in sorted(runs, reverse=True):
        event_paths = sorted(
            (run_path / "train").glob("events.out.tfevents.*")
        )
        if event_paths:
            return run_path.name, event_paths
    return "", []


def discover_experiments(source_root: Path) -> List[Experiment]:
    matched_paths = []
    for path in sorted(source_root.iterdir()):
        if path.is_dir():
            match = EXPERIMENT_RE.match(path.name)
            if match:
                matched_paths.append((path, match))

    # Older remote runs used task_size_max in the folder name (0.6, 1.0,
    # 1.6, 2.1, 2.6).  Newer sweep scripts use the requested average-size
    # label (0.4, 0.8, 1.2, 1.6, 2.0).  Detect the latter convention from
    # labels that cannot occur in the older upper-bound sequence.
    task_labels = {
        float(match.group("task_upper"))
        for path, match in matched_paths
        if int(match.group("usv")) == DEFAULT_USV
        and int(match.group("uav_res")) == DEFAULT_UAV_RES
    }
    uses_average_labels = bool(task_labels & {0.4, 0.8, 1.2, 2.0})

    experiments: List[Experiment] = []
    for path, match in matched_paths:
        upper = float(match.group("task_upper"))
        average = (
            upper
            if uses_average_labels and upper in TASK_AVERAGE_TO_RANGE
            else TASK_UPPER_TO_AVERAGE.get(upper)
        )
        if average is None:
            print(f"SKIP {path.name}: unknown task-size upper bound {upper}")
            continue
        experiments.append(
            Experiment(
                path=path,
                usv=int(match.group("usv")),
                task_upper=upper,
                task_average=average,
                uav_res=int(match.group("uav_res")),
                seed=int(match.group("seed")),
            )
        )
    return experiments


def _output_specs(experiment: Experiment) -> Iterator[Tuple[str, str, str]]:
    if (
        experiment.usv == DEFAULT_USV
        and experiment.uav_res == DEFAULT_UAV_RES
    ):
        label = f"{experiment.task_average:.1f}"
        yield (
            "task_size",
            f"task_size_{label}MB",
            f"task_size_{label}MB",
        )
    if (
        experiment.usv == DEFAULT_USV
        and experiment.task_average == DEFAULT_TASK_AVERAGE
    ):
        yield (
            "uav_res",
            f"uav_res_{experiment.uav_res}",
            f"uav_res_{float(experiment.uav_res):.1f}",
        )
    if (
        experiment.uav_res == DEFAULT_UAV_RES
        and experiment.task_average == DEFAULT_TASK_AVERAGE
    ):
        yield (
            "usv_num",
            f"usv_num_{experiment.usv}",
            f"usv_num_{experiment.usv}",
        )


def _header(experiment: Experiment, metric: str) -> str:
    task_min, task_max = TASK_AVERAGE_TO_RANGE[experiment.task_average]
    return "\n".join(
        [
            f"# USV Number: {experiment.usv}",
            "# UAV Number: 4",
            (
                f"# UAV Resource: {float(experiment.uav_res):.1f} GHz "
                f"({experiment.uav_res * 1_000_000_000} cycles/s)"
            ),
            "# Algorithm: multihop_rmappo",
            f"# Average Task Size: {experiment.task_average:.1f} MB",
            f"# Task Size Range: {task_min:.1f}-{task_max:.1f} MB",
            f"# Metric: {metric}",
            "# Format: One value per line (each line = one logged training update)",
            "# ==========================================",
        ]
    )


def write_leaf(
    experiment: Experiment,
    scalars: Mapping[str, Sequence[Tuple[int, float]]],
    leaf_dir: Path,
    filename_suffix: str,
    compute_means_source: Path | None,
) -> Tuple[int, List[str]]:
    leaf_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    missing: List[str] = []
    means: List[Tuple[str, float]] = []
    for tag, metric in TAG_TO_METRIC.items():
        points = scalars.get(tag)
        if not points:
            missing.append(tag)
            continue
        filename = f"{metric}_{filename_suffix}.txt"
        values = [value for _, value in points]
        body = "\n".join(f"{value:.6f}" for value in values)
        (leaf_dir / filename).write_text(
            _header(experiment, metric) + "\n" + body + "\n",
            encoding="utf-8",
        )
        means.append((filename, sum(values) / len(values)))
        written += 1
    summary_lines = [
        f"# 目录: {leaf_dir}",
        "# 文件均值汇总（仅计算纯数字行）",
    ]
    summary_lines.extend(
        f"{filename} 平均值: {mean:.6f}" for filename, mean in sorted(means)
    )
    (leaf_dir / "means_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    if compute_means_source and compute_means_source.is_file():
        shutil.copy2(compute_means_source, leaf_dir / "compute_means.py")
    return written, missing


def export(
    source_root: Path,
    target_root: Path,
    algorithm_dir: str,
    experiment_names: Sequence[str] | None = None,
) -> int:
    experiments = discover_experiments(source_root)
    if experiment_names:
        selected = set(experiment_names)
        experiments = [
            experiment
            for experiment in experiments
            if experiment.path.name in selected
        ]
    if not experiments:
        print(f"No sweep experiments found below {source_root}")
        return 1

    compute_means_source = (
        target_root
        / "01_lngru_ppo"
        / "task_size"
        / "task_size_0.8MB"
        / "compute_means.py"
    )
    algorithm_root = target_root / algorithm_dir
    exported_specs = 0
    for experiment in experiments:
        run_name, event_paths = find_latest_run_events(experiment.path)
        if not event_paths:
            print(f"SKIP {experiment.path.name}: no event file")
            continue
        scalars = read_scalars(event_paths)
        relevant_tags = sorted(set(scalars) & set(TAG_TO_METRIC))
        if not relevant_tags:
            print(
                f"SKIP {experiment.path.name}: no comparison scalar tags; "
                f"found {sorted(scalars)}"
            )
            continue
        for sweep, leaf_name, suffix in _output_specs(experiment):
            leaf_dir = algorithm_root / sweep / leaf_name
            count, missing = write_leaf(
                experiment,
                scalars,
                leaf_dir,
                suffix,
                compute_means_source,
            )
            exported_specs += 1
            missing_text = f"; missing={','.join(missing)}" if missing else ""
            print(
                f"EXPORT {experiment.path.name}/{run_name} -> "
                f"{leaf_dir} ({count} metrics{missing_text})"
            )

    expected = {
        "task_size": {f"task_size_{value:.1f}MB" for value in (0.4, 0.8, 1.2, 1.6, 2.0)},
        "uav_res": {f"uav_res_{value}" for value in (5, 15, 25, 35, 45)},
        "usv_num": {f"usv_num_{value}" for value in (10, 20, 30, 40, 50)},
    }
    for sweep, expected_leaves in expected.items():
        sweep_root = algorithm_root / sweep
        actual = (
            {path.name for path in sweep_root.iterdir() if path.is_dir()}
            if sweep_root.is_dir()
            else set()
        )
        missing = sorted(expected_leaves - actual)
        if missing:
            print(f"MISSING {sweep}: {', '.join(missing)}")
        else:
            print(f"COMPLETE {sweep}: {len(expected_leaves)} parameter points")
    return 0 if exported_specs else 1


def inspect(source_root: Path) -> int:
    experiments = discover_experiments(source_root)
    if not experiments:
        print(f"No sweep experiments found below {source_root}")
        return 1
    for experiment in experiments:
        run_name, event_paths = find_latest_run_events(experiment.path)
        if not event_paths:
            print(f"{experiment.path.name}: no event file")
            continue
        scalars = read_scalars(event_paths)
        print(f"{experiment.path.name}/{run_name}")
        for tag in sorted(scalars):
            points = scalars[tag]
            print(
                f"  {tag}: count={len(points)}, "
                f"step={points[0][0]}..{points[-1][0]}, "
                f"last={points[-1][1]:.6f}"
            )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=False,
        help="Path ending in results/MultihopUAVBS/MyEnv/rmappo",
    )
    parser.add_argument(
        "--event",
        type=Path,
        help="Inspect one TensorBoard event file directly",
    )
    parser.add_argument(
        "--target",
        type=Path,
        help="Path ending in comparision_results (required unless --inspect)",
    )
    parser.add_argument(
        "--algorithm-dir",
        default="06_multihop_rmappo",
        help="New algorithm folder name below the comparison root",
    )
    parser.add_argument(
        "--experiment-name",
        action="append",
        help="Only export this experiment folder (may be supplied repeatedly)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Only print scalar tags; do not write comparison files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.event is not None:
        event_path = args.event.resolve()
        scalars = read_scalars([event_path])
        print(event_path)
        for tag in sorted(scalars):
            points = scalars[tag]
            print(
                f"{tag}: count={len(points)}, "
                f"step={points[0][0]}..{points[-1][0]}, "
                f"first={points[0][1]:.6f}, last={points[-1][1]:.6f}"
            )
        return 0
    if args.source is None:
        raise SystemExit("--source is required unless --event is used")
    source = args.source.resolve()
    if args.inspect:
        return inspect(source)
    if args.target is None:
        raise SystemExit("--target is required unless --inspect is used")
    return export(
        source,
        args.target.resolve(),
        args.algorithm_dir,
        args.experiment_name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
