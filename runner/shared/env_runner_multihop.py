"""Shared-policy runner that persists multihop-specific diagnostics."""

import time

import numpy as np

from envs.envs_multihop_bs import MultihopBase
from runner.shared.env_runner import EnvRunner as ExistingEnvRunner


MULTIHOP_METRIC_KEYS = (
    "system_time",
    "total_energy",
    "avg_uav_energy",
    "avg_usv_energy",
    "completion_rate",
    "avg_uav_fly_energy",
    "avg_uav_comp_energy",
    "avg_uav_relay_energy",
    "bs_offloading_ratio",
    "route_availability_ratio",
    "avg_hop_count",
    "max_hop_count",
    "potential_passes",
    "potential_converged",
)


class EnvRunner(ExistingEnvRunner):
    """Retain the original runner behavior while logging relay diagnostics."""

    def __init__(self, config):
        super().__init__(config)
        # The parent runner reads the original experiment's base parameters.
        # Replace them for trajectory labels and multihop-only diagnostics.
        self.base = MultihopBase()
        self.num_usv = self.base.num_usv

    def run(self):
        self.warmup()
        start = time.time()
        episodes = (
            int(self.num_env_steps)
            // self.episode_length
            // self.n_rollout_threads
        )

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            episode_log = {"rewards": []}
            episode_log.update({key: [] for key in MULTIHOP_METRIC_KEYS})
            for uav_idx in range(self.num_agents):
                episode_log[f"uav_{uav_idx}_forwarded_tasks"] = []

            for step in range(self.episode_length):
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)
                obs, rewards, dones, infos = self.envs.step(actions_env)
                info_dicts = list(infos)
                trajectories = info_dicts[0].get("trajectories")
                if trajectories is not None:
                    self.plot_and_save_trajectory(episode, trajectories)

                episode_log["rewards"].append(float(np.mean(rewards)))
                for key in MULTIHOP_METRIC_KEYS:
                    metric_values = [
                        info.get(key, 0) for info in info_dicts
                    ]
                    aggregate = (
                        max(metric_values)
                        if key == "max_hop_count"
                        else np.mean(metric_values)
                    )
                    episode_log[key].append(float(aggregate))
                forwarded = np.asarray(
                    [
                        info.get(
                            "uav_forwarded_tasks",
                            [0] * self.num_agents,
                        )
                        for info in info_dicts
                    ],
                    dtype=float,
                )
                for uav_idx in range(self.num_agents):
                    episode_log[f"uav_{uav_idx}_forwarded_tasks"].append(
                        float(np.mean(forwarded[:, uav_idx]))
                    )

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                self.insert(data)

            self.compute()
            train_infos = self.train()
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode % self.log_interval == 0:
                elapsed = time.time() - start
                metrics = {
                    "episode_reward": float(np.mean(episode_log["rewards"]))
                }
                for key, metric_values in episode_log.items():
                    if key == "rewards":
                        continue
                    aggregate = (
                        max(metric_values)
                        if key == "max_hop_count"
                        else np.mean(metric_values)
                    )
                    metrics[key] = float(aggregate)
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, "
                    "total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / elapsed),
                    )
                )
                print(f"--- Episode {episode} Multihop Performance Summary ---")
                print(f"  Avg. Reward: {metrics['episode_reward']:.3f}")
                print(f"  Avg. System Time: {metrics['system_time']:.3f} s")
                print(
                    f"  Avg. Completion Rate: {metrics['completion_rate']:.2f} %"
                )
                print(
                    "  Route Availability / BS Offloading: "
                    f"{metrics['route_availability_ratio']:.2f} % / "
                    f"{metrics['bs_offloading_ratio']:.2f} %"
                )
                print(
                    "  Avg. / Max Wireless Hops: "
                    f"{metrics['avg_hop_count']:.2f} / "
                    f"{metrics['max_hop_count']:.2f}"
                )
                print(
                    "  Avg. UAV Relay Energy: "
                    f"{metrics['avg_uav_relay_energy']:.3f} J"
                )
                print("-----------------------------------------")
                self.log_train(metrics, total_num_steps)
                self.log_train(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
