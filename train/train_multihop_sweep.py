"""Parametrized training entry point for multihop UAV-BS comparison experiments.

Supports command-line overrides for num_usv, task_size, and uav_resource
so that a single script can drive all comparison sweeps without duplicating
train/env files.
"""

import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from envs.env_continuous_multihop import ContinuousMultihopEnv
from envs.env_wrappers import DummyVecEnv
from envs.envs_multihop_bs import MultihopBase


def _build_env_overrides(all_args):
    """Construct the overrides dict from parsed command-line arguments."""
    MB = 8 * 1024 * 1024
    GHz = 1_000_000_000
    overrides = {}
    if all_args.num_usv is not None:
        overrides["num_usv"] = all_args.num_usv
    if all_args.task_size_max is not None:
        overrides["task_size_max"] = all_args.task_size_max * MB
    if all_args.task_size_min is not None:
        overrides["task_size_min"] = all_args.task_size_min * MB
    if all_args.uav_resource is not None:
        overrides["uav_resource"] = all_args.uav_resource * GHz
    return overrides or None


def _auto_experiment_name(all_args, overrides):
    """Generate a descriptive experiment name from the sweep parameters."""
    if all_args.experiment_name != "check":
        return all_args.experiment_name
    MB = 8 * 1024 * 1024
    GHz = 1_000_000_000
    base = MultihopBase(overrides)
    usv = base.num_usv
    task = (
        all_args.task_size_label
        if all_args.task_size_label is not None
        else base.task_size_max / MB
    )
    res = int(base.uav_resource / GHz)
    return f"usv{usv}_task{task:.1f}M_res{res}G_seed{all_args.seed}"


def make_train_env(all_args, env_overrides):
    def get_env_fn(rank):
        def init_env():
            env = ContinuousMultihopEnv(env_overrides=env_overrides)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    return DummyVecEnv(
        [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
    )


def make_eval_env(all_args, env_overrides):
    def get_env_fn(rank):
        def init_env():
            env = ContinuousMultihopEnv(env_overrides=env_overrides)
            env.seed(all_args.seed + 500_000 + rank * 1000)
            return env
        return init_env

    return DummyVecEnv(
        [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
    )


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_usv", type=int, default=None,
                        help="Override USV count (default: use MultihopBase default)")
    parser.add_argument("--task_size_max", type=float, default=None,
                        help="Override task_size_max in MB (e.g. 1.0)")
    parser.add_argument("--task_size_min", type=float, default=None,
                        help="Override task_size_min in MB (e.g. 0.5)")
    parser.add_argument("--task_size_label", type=float, default=None,
                        help="Task-size label used in the experiment name")
    parser.add_argument("--uav_resource", type=float, default=None,
                        help="Override UAV resource in GHz (e.g. 25)")
    parser.set_defaults(
        env_name="MultihopUAVBS",
        use_linear_lr_decay=False,
    )
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    env_overrides = _build_env_overrides(all_args)
    all_args.env_overrides = env_overrides

    base = MultihopBase(env_overrides)
    all_args.num_agents = base.num_uav
    all_args.episode_length = base.episode_length
    all_args.share_policy = True

    all_args.experiment_name = _auto_experiment_name(all_args, env_overrides)

    if all_args.algorithm_name == "rmappo":
        assert (
            all_args.use_recurrent_policy
            or all_args.use_naive_recurrent_policy
        ), "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert not all_args.use_recurrent_policy
        assert not all_args.use_naive_recurrent_policy
    else:
        raise NotImplementedError

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = (
        PROJECT_ROOT / "results"
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = [
        int(folder.name.split("run")[1])
        for folder in run_dir.iterdir()
        if folder.name.startswith("run") and folder.name[3:].isdigit()
    ]
    current_run = f"run{max(existing_runs) + 1}" if existing_runs else "run1"
    run_dir = run_dir / current_run
    run_dir.mkdir(parents=True, exist_ok=True)

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}-"
        f"{all_args.experiment_name}@{all_args.user_name}"
    )
    print(
        f"Starting multihop sweep experiment on "
        f"{socket.gethostname()} in {run_dir} "
        f"({base.num_uav} UAVs, {base.num_usv} USVs)"
    )
    if env_overrides:
        print(f"  Environment overrides: {env_overrides}")

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args, env_overrides)
    eval_envs = make_eval_env(all_args, env_overrides) if all_args.use_eval else None
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    from runner.shared.env_runner_multihop import EnvRunner as Runner

    runner = Runner(config)
    runner.run()
    envs.close()
    if eval_envs is not None and eval_envs is not envs:
        eval_envs.close()
    if hasattr(runner.writter, "export_scalars_to_json"):
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + "/summary.json")
        )
    else:
        runner.writter.flush()
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
