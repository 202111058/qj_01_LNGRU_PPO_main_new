"""Independent training entry point for the multihop UAV-BS experiment."""

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


ENV_BASE = MultihopBase()


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = ContinuousMultihopEnv()
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv(
        [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
    )


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = ContinuousMultihopEnv()
            env.seed(all_args.seed + 500_000 + rank * 1000)
            return env

        return init_env

    return DummyVecEnv(
        [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
    )


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument(
        "--num_agents", type=int, default=ENV_BASE.num_uav
    )
    parser.set_defaults(env_name="MultihopUAVBS")
    return parser.parse_known_args(args)[0]


def _validate_fixed_args(all_args):
    """Protect the experiment namespace and core/wrapper shape invariants."""

    expected = {
        "env_name": "MultihopUAVBS",
        "num_agents": ENV_BASE.num_uav,
        "episode_length": ENV_BASE.episode_length,
        "share_policy": True,
    }
    invalid = {
        name: (getattr(all_args, name), value)
        for name, value in expected.items()
        if getattr(all_args, name) != value
    }
    if invalid:
        detail = ", ".join(
            f"{name}={actual!r} (required {required!r})"
            for name, (actual, required) in invalid.items()
        )
        raise ValueError(f"Invalid fixed multihop arguments: {detail}")


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    _validate_fixed_args(all_args)

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
        "Starting multihop UAV-BS experiment on "
        f"{socket.gethostname()} in {run_dir} "
        f"({ENV_BASE.num_uav} UAVs, {ENV_BASE.num_usv} USVs)"
    )

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
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
