import time
import numpy as np
import torch
from runner.shared.base_runner import Runner
import matplotlib.pyplot as plt
import os
from envs.base import base


def _t2n(x):
    """将torch tensor转换为numpy array"""
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation, and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

        # 创建轨迹保存目录
        self.trajectory_dir = str(self.run_dir / 'trajectories')
        if not os.path.exists(self.trajectory_dir):
            os.makedirs(self.trajectory_dir)

        # 初始化base参数
        self.base = base()

        # 为UAV定义一个颜色列表
        self.uav_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']

        # ============ 创建指标保存目录 ============
        self.metrics_dir = str(self.run_dir / 'metrics')
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)

        # 从base中读取USV数量
        self.num_usv = self.base.num_usv

        # 初始化指标文件(使用USV数量标识)
        self.metric_files = {
            'avg_reward': open(f"{self.metrics_dir}/avg_reward_usv_num_{self.num_usv}.txt", 'w'),
            'avg_system_time': open(f"{self.metrics_dir}/avg_system_time_usv_num_{self.num_usv}.txt", 'w'),
            'avg_total_energy': open(f"{self.metrics_dir}/avg_total_energy_usv_num_{self.num_usv}.txt", 'w'),
            'avg_uav_energy': open(f"{self.metrics_dir}/avg_uav_energy_usv_num_{self.num_usv}.txt", 'w'),
            'avg_uav_fly_energy': open(f"{self.metrics_dir}/avg_uav_fly_energy_usv_num_{self.num_usv}.txt", 'w'),
            'avg_uav_comp_energy': open(f"{self.metrics_dir}/avg_uav_comp_energy_usv_num_{self.num_usv}.txt", 'w'),
            'avg_usv_energy': open(f"{self.metrics_dir}/avg_usv_energy_usv_num_{self.num_usv}.txt", 'w'),
            'avg_completion_rate': open(f"{self.metrics_dir}/avg_completion_rate_usv_num_{self.num_usv}.txt", 'w'),
        }

        # 写入文件头(USV数量信息)
        for key, file in self.metric_files.items():
            file.write(f"# USV Number: {self.num_usv}\n")
            file.write(f"# Metric: {key}\n")
            file.write(f"# Format: One value per line (each line = one episode)\n")
            file.write("# ==========================================\n")

        # ============ 新增：指标缓冲机制 ============
        self.metric_buffer = {
            'avg_reward': [],
            'avg_system_time': [],
            'avg_total_energy': [],
            'avg_uav_energy': [],
            'avg_uav_fly_energy': [],
            'avg_uav_comp_energy': [],
            'avg_usv_energy': [],
            'avg_completion_rate': []
        }

        # 缓冲区配置
        self.buffer_flush_interval = 50  # 每50个episode写入一次
        self.trajectory_save_interval = 100  # 每100个episode保存一次轨迹图
        # ===============================================

        print(f"\n{'=' * 60}")
        print(f"EnvRunner Initialized")
        print(f"  USV Count: {self.num_usv}")
        print(f"  Metrics Dir: {self.metrics_dir}")
        print(f"  Trajectory Dir: {self.trajectory_dir}")
        print(f"  Trajectory Save Interval: {self.trajectory_save_interval} episodes")
        print(f"  Metric Flush Interval: {self.buffer_flush_interval} episodes")
        print(f"{'=' * 60}\n")

    def __del__(self):
        """析构函数:关闭所有打开的文件"""
        if hasattr(self, 'metric_files'):
            for file in self.metric_files.values():
                try:
                    file.close()
                except:
                    pass

    def flush_metrics_to_disk(self):
        """将缓冲区的指标批量写入磁盘"""
        if not self.metric_buffer['avg_reward']:  # 如果缓冲区为空，直接返回
            return

        for key in self.metric_files.keys():
            for val in self.metric_buffer[key]:
                self.metric_files[key].write(f"{val:.6f}\n")
            self.metric_buffer[key].clear()

        # 批量写入后才flush
        for file in self.metric_files.values():
            file.flush()

    def plot_and_save_trajectory(self, episode, trajectories):
        """
        绘制并保存在一个episode中所有UAV和USV的轨迹。

        Args:
            episode: 当前episode编号
            trajectories: 包含UAV和USV轨迹的字典
        """
        if trajectories is None:
            print(f"Warning: No trajectory data to plot for episode {episode}.")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # --- 绘制UAV轨迹 (每个UAV使用不同颜色) ---
        for i, traj in enumerate(trajectories['uavs']):
            traj_np = np.array(traj)
            color = self.uav_colors[i % len(self.uav_colors)]
            ax.plot(traj_np[:, 0], traj_np[:, 1], marker='.', linestyle='-',
                    color=color, label=f'UAV {i} Trajectory', linewidth=2, markersize=4)
            ax.plot(traj_np[0, 0], traj_np[0, 1], marker='o', color=color,
                    markersize=10, label=f'UAV {i} Start')
            ax.plot(traj_np[-1, 0], traj_np[-1, 1], marker='*', color=color,
                    markersize=15, markeredgecolor='black', label=f'UAV {i} End')

        # --- 绘制USV轨迹 (所有USV使用相同颜色和图例) ---
        for i, traj in enumerate(trajectories['usvs']):
            traj_np = np.array(traj)
            if i == 0:
                ax.plot(traj_np[:, 0], traj_np[:, 1], marker='x', markersize=2,
                        linestyle=':', color='gray', alpha=0.9, label='USV Trajectory')
            else:
                ax.plot(traj_np[:, 0], traj_np[:, 1], marker='x', markersize=2,
                        linestyle=':', color='gray', alpha=0.9)

        ax.set_xlabel("X Coordinate (m)", fontsize=12)
        ax.set_ylabel("Y Coordinate (m)", fontsize=12)
        ax.set_title(f"Episode {episode} Trajectories (USV Num: {self.num_usv})", fontsize=14)
        ax.set_xlim(self.base.field_X)
        ax.set_ylim(self.base.field_Y)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

        # 保存图片，降低DPI提高速度
        save_path = f"{self.trajectory_dir}/episode_{episode}_usv_{self.num_usv}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')  # 从150降到100
        plt.close(fig)
        print(f"  [Trajectory Saved] Episode {episode}")

    def run(self):
        """主训练循环"""
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        print(f"\n{'=' * 60}")
        print(f"Starting Training: {episodes} episodes")
        print(f"{'=' * 60}\n")

        # 用于临时存储每个episode的轨迹数据
        latest_trajectories = None

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # 初始化用于记录整个episode指标的列表
            episode_log = {
                'rewards': [],
                'system_time': [],
                'total_energy': [],
                'avg_uav_energy': [],
                'avg_usv_energy': [],
                'completion_rate': [],
                'avg_uav_fly_energy': [],
                'avg_uav_comp_energy': []
            }

            # Episode内的step循环
            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Observe reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                info = infos[0]

                # ========== 优化：只保存轨迹数据，不立即绘制 ==========
                if "trajectories" in info:
                    latest_trajectories = info.pop('trajectories')

                # 记录step级别的指标
                episode_log['rewards'].append(rewards[0][0])
                for key in episode_log:
                    if key != 'rewards':
                        episode_log[key].append(info.get(key, 0))

                # 存储数据到buffer
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

            # Episode结束后计算advantage
            self.compute()

            # 训练网络
            train_infos = self.train()

            # 计算总步数
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # ========== 每100个episode保存一次轨迹图 ==========
            if episode % self.trajectory_save_interval == 0 and latest_trajectories is not None:
                self.plot_and_save_trajectory(episode, latest_trajectories)

            # 定期保存模型
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # 定期记录和打印日志
            if episode % self.log_interval == 0:
                end = time.time()

                # ========== 1. 计算本回合的平均指标 ==========
                avg_ep_reward = np.mean(episode_log['rewards'])
                avg_ep_time = np.mean(episode_log['system_time'])
                avg_ep_total_energy = np.mean(episode_log['total_energy'])
                avg_ep_uav_energy = np.mean(episode_log['avg_uav_energy'])
                avg_ep_usv_energy = np.mean(episode_log['avg_usv_energy'])
                avg_ep_completion_rate = np.mean(episode_log['completion_rate'])
                avg_ep_uav_fly_energy = np.mean(episode_log['avg_uav_fly_energy'])
                avg_ep_uav_comp_energy = np.mean(episode_log['avg_uav_comp_energy'])

                # ========== 2. 添加到缓冲区 ==========
                self.metric_buffer['avg_reward'].append(avg_ep_reward)
                self.metric_buffer['avg_system_time'].append(avg_ep_time)
                self.metric_buffer['avg_total_energy'].append(avg_ep_total_energy)
                self.metric_buffer['avg_uav_energy'].append(avg_ep_uav_energy)
                self.metric_buffer['avg_uav_fly_energy'].append(avg_ep_uav_fly_energy)
                self.metric_buffer['avg_uav_comp_energy'].append(avg_ep_uav_comp_energy)
                self.metric_buffer['avg_usv_energy'].append(avg_ep_usv_energy)
                self.metric_buffer['avg_completion_rate'].append(avg_ep_completion_rate)

                # ========== 3. 定期批量写入磁盘 ==========
                if len(self.metric_buffer['avg_reward']) >= self.buffer_flush_interval:
                    self.flush_metrics_to_disk()
                    print(f"  [Metrics Flushed] Episode {episode} - Buffer written to disk")

                # ========== 4. 打印性能摘要到控制台 ==========
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                print(f"{'=' * 60}")
                print(f"Episode {episode} Performance Summary (USV Num: {self.num_usv})")
                print(f"{'=' * 60}")
                print(f"  Avg. Reward:            {avg_ep_reward:>10.3f}")
                print(f"  Avg. System Time:       {avg_ep_time:>10.3f} s")
                print(f"  Avg. Total Energy:      {avg_ep_total_energy:>10.3f} J")
                print(f"  Avg. UAV Energy:        {avg_ep_uav_energy:>10.3f} J")
                print(f"    - Avg. UAV Fly Energy:  {avg_ep_uav_fly_energy:>8.3f} J")
                print(f"    - Avg. UAV Comp Energy: {avg_ep_uav_comp_energy:>8.3f} J")
                print(f"  Avg. USV Energy:        {avg_ep_usv_energy:>10.3f} J")
                print(f"  Avg. Completion Rate:   {avg_ep_completion_rate:>10.2f} %")
                print(f"  Buffer Size:            {len(self.metric_buffer['avg_reward'])} episodes")
                print(f"{'=' * 60}\n")

                # ========== 5. 记录到TensorBoard ==========
                metrics_to_log = {
                    'episode_reward': avg_ep_reward,
                    'system_time': avg_ep_time,
                    'total_energy': avg_ep_total_energy,
                    'avg_uav_energy': avg_ep_uav_energy,
                    'avg_usv_energy': avg_ep_usv_energy,
                    'completion_rate': avg_ep_completion_rate,
                    'avg_uav_fly_energy': avg_ep_uav_fly_energy,
                    'avg_uav_comp_energy': avg_ep_uav_comp_energy
                }

                self.log_train(metrics_to_log, total_num_steps)

            # 定期评估
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        # ========== 训练结束后处理 ==========
        # 1. 写入缓冲区剩余数据
        self.flush_metrics_to_disk()
        print(f"\n[Final Flush] All remaining metrics written to disk")

        # 2. 保存最后一个episode的轨迹图
        if latest_trajectories is not None:
            self.plot_and_save_trajectory(episodes - 1, latest_trajectories)
            print(f"[Final Trajectory] Episode {episodes - 1} trajectory saved")

        # 3. 关闭所有文件
        print(f"\n{'=' * 60}")
        print(f"Training Completed!")
        print(f"Total Episodes: {episodes}")
        print(f"Closing metric files...")
        print(f"{'=' * 60}\n")

        for key, file in self.metric_files.items():
            try:
                file.close()
                print(f"  Closed: {key}_usv_num_{self.num_usv}.txt")
            except:
                pass

    def warmup(self):
        """预热环境，重置并收集初始观测"""
        obs = self.envs.reset()
        share_obs = obs.reshape(self.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        """收集训练数据"""
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions

    def insert(self, data):
        """将数据插入到buffer中"""
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = obs.reshape(self.n_rollout_threads, -1)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        """评估当前策略"""
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                *self.buffer.rnn_states.shape[2:],
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions
            )
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(
            np.array(eval_episode_rewards), axis=0
        )
        eval_average_episode_rewards = np.mean(
            eval_env_infos["eval_average_episode_rewards"]
        )
        print(f"Eval average episode rewards: {eval_average_episode_rewards}")
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """渲染环境（如果支持）"""
        envs = self.envs
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(
                    np.split(_t2n(rnn_states), self.n_rollout_threads)
                )

                obs, rewards, dones, infos = envs.step(actions)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    (
                        (dones == True).sum(),
                        self.recurrent_N,
                        self.hidden_size,
                    ),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

            print(
                f"Episode {episode}, average episode reward: {np.mean(np.sum(np.array(episode_rewards), axis=0))}"
            )

        if self.all_args.save_gifs:
            import imageio
            imageio.mimsave(
                str(self.gif_dir) + "/render.gif",
                all_frames,
                duration=self.all_args.ifi,
            )
