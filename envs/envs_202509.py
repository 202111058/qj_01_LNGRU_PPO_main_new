# envs_202509.py
# 作用：实现强化学习环境的核心逻辑，包括状态、动作、奖励和环境演进。
# 这是对 env_core_202407.py 的完整重构。

import numpy as np
from .base import base
from .Init_player import init_usv, init_uav, init_leo
from .Opi_RA_TO import PotentialGame
from .common import common



# 假设您的博弈决策逻辑在一个名为Opi_RA_TO.py的文件中
# from .Opi_RA_TO import PotentialGame

class EnvCore(object):
    """
    一个多智能体强化学习环境，用于模拟UAV辅助下的海上移动边缘计算网络。

    - 智能体 (Agents): 无人机 (UAVs)
    - 任务源 (Task Sources): 无人水面艇 (USVs)
    - 目标 (Objective): 联合优化UAV的轨迹，以最小化所有USV任务的完成时间和系统总能耗。
    """

    def __init__(self):
        self.base = base()
        self.common = common(self.base)

        self.num_usvs = self.base.num_usv
        self.num_uavs = self.base.num_uav

        # 初始化设备列表
        self.usvs = []
        self.uavs = []
        self.leo = None
        self.usv_uav_channel_snapshot = None

        # 卸载决策向量，在reset时由博弈算法决定
        self.offloading_decisions = np.zeros(self.num_usvs, dtype=int)

        # 状态与动作空间维度
        # 每个UAV的观测：自身位置(2) + 所有USV位置(2*N) + 所有USV任务(1*N) = 2 + 1*N
        self.obs_dim = 2 + 4 * self.num_usvs
        self.action_dim = 2  # UAV的x, y方向速度

        # 新增一个内部变量来记录当前步数
        self.current_step = 0


        # 初始化每个step需要计算的性能指标
        self.system_time = 0.0  # 系统完成所有任务所需的最大时间 (ms)
        self.total_energy = 0.0  # 系统总能耗 (J)
        self.avg_usv_energy = 0.0  # USV平均能耗 (J)
        self.avg_uav_energy = 0.0  # UAV平均能耗 (J)
        self.task_completion_rate = 0.0  # 任务完成率 (%)

    def reset(self):
        """环境重置，返回初始观测状态"""
        self.current_step = 0  # 重置步数计数器

        # 1. 初始化所有实体
        self.usvs = init_usv(self.base)
        self.uavs = init_uav(self.base)
        self.leo = init_leo(self.base)
        self.usv_uav_channel_snapshot = (
            self.common.build_usv_to_uav_channel_snapshot(
                self.usvs, self.uavs
            )
        )


        # 2. 初始卸载决策 (这里可以调用您的博弈算法)
        # 示例：随机初始化决策
        for i in range(self.num_usvs):
            self.offloading_decisions[i] = self.usvs[i]['offload_decision']
        self.offloading_decisions = PotentialGame(
            self.usvs,
            self.uavs,
            self.leo,
            self.base,
            self.offloading_decisions,
            self.usv_uav_channel_snapshot,
        )

        # 3. 返回初始的联合观测
        return self._get_observation()

    def normalize_obj(self, obj, min_val, max_val):
        """
        通用的Min-Max归一化函数，将数据缩放到[0, 1]区间。
        增加了除零保护。
        """
        range_val = max_val - min_val
        if range_val == 0:
            return 0.5  # 如果最大最小值相等，返回一个中性值
        normalized_obj = (obj - min_val) / range_val
        return normalized_obj

    def _get_observation(self):
        """为每个UAV智能体构建其观测向量 (调用normalize_obj)"""
        joint_obs = []
        for uav in self.uavs:
            obs = []
            # 1. 归一化的自身位置 (2维)
            norm_x = self.normalize_obj(uav['position'][0], self.base.field_X[0], self.base.field_X[1])
            norm_y = self.normalize_obj(uav['position'][1], self.base.field_Y[0], self.base.field_Y[1])
            obs.extend([norm_x, norm_y])

            # 2. 归一化的所有USV的位置和任务信息
            for usv in self.usvs:
                # 位置归一化
                norm_usv_x = self.normalize_obj(usv['position'][0], self.base.field_X[0], self.base.field_X[1])
                norm_usv_y = self.normalize_obj(usv['position'][1], self.base.field_Y[0], self.base.field_Y[1])
                obs.extend([norm_usv_x, norm_usv_y])

                # 任务大小归一化
                task_size_norm = self.normalize_obj(usv['task_size'], self.base.task_size_min, self.base.task_size_max)
                obs.append(task_size_norm)

                # 任务资源归一化
                task_res_norm = self.normalize_obj(usv['task_resource'], self.base.task_resources_min,
                                                   self.base.task_resources_max)
                obs.append(task_res_norm)

            joint_obs.append(np.array(obs, dtype=np.float32))
        return joint_obs

    def calculate_rewards(self, task_total_times, task_completion_rate, uav_comp_energies, uav_fly_energies):
        """
        计算每个UAV的奖励，包含所有重要的奖励组件 (您设计的版本)
        """
        rewards = np.zeros(self.num_uavs)

        # 只要完成率达到阈值，就给予奖励
        completion_reward = 0
        if task_completion_rate >= self.base.completion_threshold:
            base_reward = 10.0  # 增加基础奖励
            normalized_rate = (task_completion_rate - self.base.completion_threshold) / (
                        100 - self.base.completion_threshold)
            bonus_reward = 30.0 * normalized_rate  # 增加额外奖励
            completion_reward = base_reward + bonus_reward
        else:
            # 即使未达到阈值也给予一定奖励，鼓励进步
            completion_reward = 30.0 * (task_completion_rate / self.base.completion_threshold)


        for k in range(self.num_uavs):
            # --- 1. 时延惩罚 ---
            # assigned_tasks = np.where(self.offloading_decisions == k + 1)[0]
            # if len(assigned_tasks) > 0:
            #     delay_penalty = -np.mean(task_total_times[assigned_tasks])
            # else:
            delay_penalty = -self.system_time

            # --- 2. 能耗惩罚 ---
            energy_penalty = 0
            uav_total_energy = uav_comp_energies[k] + uav_fly_energies[k]
            if uav_total_energy > self.base.energy_threshold:
                excess_ratio = min(2.0, uav_total_energy / self.base.energy_threshold) - 1.0
                energy_penalty = -20.0 * excess_ratio

            # --- 3. 边界惩罚 ---
            boundary_penalty = 0
            uav_pos = self.uavs[k]['position']
            if (uav_pos[0] <= self.base.field_X[0] or uav_pos[0] >= self.base.field_X[1] or
                    uav_pos[1] <= self.base.field_Y[0] or uav_pos[1] >= self.base.field_Y[1]):
                boundary_penalty = -100.0
            else:
                dist_x = min(uav_pos[0] - self.base.field_X[0], self.base.field_X[1] - uav_pos[0])
                dist_y = min(uav_pos[1] - self.base.field_Y[0], self.base.field_Y[1] - uav_pos[1])
                min_dist = min(dist_x, dist_y)
                if min_dist < self.base.boundary_margin:
                    boundary_penalty = -30.0 * (1.0 - min_dist / self.base.boundary_margin)

            # --- 4. 碰撞惩罚 ---
            collision_penalty = 0
            for j in range(self.num_uavs):
                if j != k:
                    distance = np.linalg.norm(self.uavs[k]['position'] - self.uavs[j]['position'])
                    if distance < self.base.collision_threshold:
                        collision_penalty -= 20.0 * (1.0 - distance / self.base.collision_threshold)

            # 新增：智能稳定性奖励
            stability_reward = self._calculate_stability_reward(k)

            # 新增：简化的覆盖奖励
            coverage_reward = self._calculate_coverage_reward(k)

            # --- 5. 综合奖励 ---
            rewards[k] = (
                    self.base.w_delay * delay_penalty +
                    self.base.w_energy * energy_penalty +
                    self.base.w_boundary * boundary_penalty +
                    self.base.w_collision * collision_penalty +
                    self.base.w_completion * completion_reward +
                    stability_reward +
                    coverage_reward
            )

        return rewards

    # 添加辅助方法计算UAV附近的USV数量
    def _calculate_stability_reward(self, uav_idx):
        """改进的稳定性奖励，更强烈地鼓励UAV在良好位置保持静止"""
        # 降低性能阈值，更早开始奖励稳定
        if self.task_completion_rate >= 75:  # 降低阈值
            if len(self.uavs[uav_idx]['trajectory']) > 1:
                last_pos = self.uavs[uav_idx]['trajectory'][-2]
                current_pos = self.uavs[uav_idx]['position']
                movement = np.linalg.norm(current_pos - last_pos)

                covered_usvs = self._count_nearby_usvs(uav_idx)
                coverage_ratio = covered_usvs / self.num_usvs

                # 根据覆盖率和系统性能动态调整稳定性奖励
                if coverage_ratio >= 0.15:  # 降低覆盖阈值
                    # 使用指数衰减函数，对小移动给予极大奖励，对大移动给予极小奖励
                    stability_base = 15.0  # 基础奖励值

                    # 根据系统性能调整奖励强度
                    if self.task_completion_rate >= 80:
                        stability_multiplier = 1.5  # 系统性能很好时，增强稳定性奖励
                    else:
                        stability_multiplier = 1.0

                    # 指数衰减函数：movement越小，reward越接近最大值
                    decay_rate = 2.0  # 控制衰减速度
                    # stability_reward = stability_base * stability_multiplier * np.exp(-decay_rate * movement)
                    stability_reward = stability_base * stability_multiplier / (1.0 + decay_rate * movement)

                    # 根据覆盖率增加奖励
                    coverage_bonus = min(1.5, coverage_ratio * 3.0)  # 覆盖率越高，奖励越高，最高1.5倍

                    return stability_reward * coverage_bonus

        return 0

    def _calculate_coverage_reward(self, uav_idx):
        """改进的覆盖奖励，平衡覆盖与移动的关系"""
        covered_usvs = self._count_nearby_usvs(uav_idx)
        coverage_ratio = covered_usvs / self.num_usvs

        if covered_usvs > 0:
            # 基础覆盖奖励：使用非线性函数，覆盖更多USV获得更高奖励
            base_coverage_reward = 15.0 * np.sqrt(covered_usvs)

            # 移动惩罚：如果覆盖率已经不错但仍在大幅移动，减少奖励
            movement_penalty = 0
            if len(self.uavs[uav_idx]['trajectory']) > 1 and coverage_ratio >= 0.3:
                last_pos = self.uavs[uav_idx]['trajectory'][-2]
                current_pos = self.uavs[uav_idx]['position']
                movement = np.linalg.norm(current_pos - last_pos)

                # 移动越大，惩罚越严重，但设置上限避免过度惩罚
                movement_penalty = min(0.2 * base_coverage_reward, 5.0 * movement)

            # 最终覆盖奖励 = 基础覆盖奖励 - 移动惩罚
            return max(0, base_coverage_reward - movement_penalty)

        return 0

    def _count_nearby_usvs(self, uav_idx):
        """计算UAV附近radius范围内的USV数量"""
        count = 0
        uav_pos = self.uavs[uav_idx]['position']
        for usv in self.usvs:
            distance = np.linalg.norm(uav_pos - usv['position'])
            if distance <= self.base.uav_coverage:
                count += 1
        return count

    def step(self, actions):
        """
        环境执行一个时间步。
        - actions: 一个列表，包含每个UAV智能体的动作。
        """
        # --- 1. 更新UAV位置并计算飞行能耗 ---
        uav_fly_energies = np.zeros(self.num_uavs)
        for k, uav in enumerate(self.uavs):
            action = actions[k]  # shape: (2,)
            old_position = uav['position'].copy()

            # 计算该UAV覆盖的USV数量
            covered_usvs = self._count_nearby_usvs(k)
            coverage_ratio = covered_usvs / self.num_usvs

            # # 使用连续函数计算动作缩放
            # # 简化版本的动作缩放
            # if self.task_completion_rate >= 90 and coverage_ratio >= 0.3:
            #     # 系统性能很好：使用非常小的最大速度，并且进一步缩小动作范围
            #     max_speed = 2.0
            #     action_scale = 0.3  # 只使用动作的30%
            # elif self.task_completion_rate >= 85 and coverage_ratio >= 0.15:
            #     # 系统性能良好：使用较小的最大速度，稍微缩小动作范围
            #     max_speed = 5.0
            #     action_scale = 0.7  # 使用动作的60%
            # elif self.task_completion_rate >= 70:
            #     # 系统性能一般：使用中等最大速度，轻微缩小动作范围
            #     max_speed = 10.0
            #     action_scale = 0.8  # 使用动作的80%
            # else:
            #     # 系统性能不佳：使用较大最大速度，不缩小动作范围
            #     max_speed = 15.0
            #     action_scale = 1.0  # 使用完整动作
            #
            # # 如果覆盖率特别高，无论系统性能如何，都强制降低速度
            # if coverage_ratio >= 0.3 and self.task_completion_rate >= 90:
            #     max_speed = min(max_speed, 2.0)
            #     action_scale = 0.2

            # if self.task_completion_rate >= 90 and coverage_ratio >= 0.2:
            #     # 系统性能很好：使用非常小的最大速度，并且进一步缩小动作范围
            #     max_speed = 2.0
            #     action_scale = 0.3  # 只使用动作的30%
            # elif self.task_completion_rate >= 85 and coverage_ratio >= 0.15:
            #     # 系统性能良好：使用较小的最大速度，稍微缩小动作范围
            #     max_speed = 5.0
            #     action_scale = 0.6  # 使用动作的60%
            # elif self.task_completion_rate >= 70:
            #     # 系统性能一般：使用中等最大速度，轻微缩小动作范围
            #     max_speed = 10.0
            #     action_scale = 0.8  # 使用动作的80%
            # else:
            #     # 系统性能不佳：使用较大最大速度，不缩小动作范围
            #     max_speed = 15.0
            #     action_scale = 1.0  # 使用完整动作
            #
            # # 如果覆盖率特别高，无论系统性能如何，都强制降低速度
            # if coverage_ratio >= 0.3:
            #     max_speed = min(max_speed, 2.0)
            #     action_scale = 0.2

            if self.task_completion_rate >= 90:
                # 系统性能很好：使用非常小的最大速度，并且进一步缩小动作范围
                max_speed = 6.0
                action_scale = 0.3  # 只使用动作的30%
            elif self.task_completion_rate >= 85:
                # 系统性能良好：使用较小的最大速度，稍微缩小动作范围
                max_speed = 9.0
                action_scale = 0.5  # 使用动作的60%
            elif self.task_completion_rate >= 70:
                # 系统性能一般：使用中等最大速度，轻微缩小动作范围
                max_speed = 11.0
                action_scale = 0.6  # 使用动作的80%
            elif self.task_completion_rate >= 60:
                # 系统性能一般：使用中等最大速度，轻微缩小动作范围
                max_speed = 13.0
                action_scale = 0.8  # 使用动作的80%
            else:
                # 系统性能不佳：使用较大最大速度，不缩小动作范围
                max_speed = 15.0
                action_scale = 1.0  # 使用完整动作

            # 如果覆盖率特别高，无论系统性能如何，都强制降低速度
            if coverage_ratio >= 0.3:
                max_speed = min(max_speed, 2.0)
                action_scale = 0.5

            # 将RL Agent输出的动作 (通常在[-1, 1]) 映射到实际速度 (m/s)
            scaled_action = action * action_scale
            velocity_vector = scaled_action * max_speed

            # 更新位置
            move_distance = velocity_vector
            uav['position'] += move_distance

            # 边界检查
            uav['position'][0] = np.clip(uav['position'][0], self.base.field_X[0], self.base.field_X[1])
            uav['position'][1] = np.clip(uav['position'][1], self.base.field_Y[0], self.base.field_Y[1])
            uav['velocity_vector'] = (
                uav['position'] - old_position
            ) / self.base.run_slot

            # 记录uav轨迹
            uav['trajectory'].append(np.copy(uav['position']))

            # 计算飞行能耗
            velocity_magnitude = np.linalg.norm(velocity_vector)
            uav_fly_energies[k] = self.common.get_uav_fly_energy(velocity_magnitude)

        self.usv_uav_channel_snapshot = (
            self.common.build_usv_to_uav_channel_snapshot(
                self.usvs, self.uavs
            )
        )
        self.offloading_decisions = PotentialGame(
            self.usvs,
            self.uavs,
            self.leo,
            self.base,
            self.offloading_decisions,
            self.usv_uav_channel_snapshot,
        )

        # --- 2. 根据卸载决策，计算每个任务的时延和能耗 ---
        task_total_times = np.zeros(self.num_usvs)
        usv_energies = np.zeros(self.num_usvs)
        uav_comp_energies = np.zeros(self.num_uavs)
        completed_tasks = 0

        # # todo
        # detailed_task_times = []  # 这个列表我们上次已经加了
        # decisions_this_step = self.offloading_decisions.copy()

        # SECTION 资源分配部分，首先计算资源分配系数
        pc = np.zeros(self.num_usvs)
        pb = np.zeros(self.num_usvs)

        # 卸载到卫星的USV
        leo_idxs = np.where(self.offloading_decisions == self.num_uavs + 1)[0].astype(int)
        if len(leo_idxs) > 0:
            pc_leo, pb_leo = self.common.get_leo_pc_pb(self.usvs, leo_idxs, self.leo, self.base)
            pc += pc_leo
            pb += pb_leo

        # 卸载到UAV的USV
        for k in range(self.num_uavs):
            uav_idxs = np.where(self.offloading_decisions == k + 1)[0].astype(int)
            if len(uav_idxs) > 0:
                pc_uav, pb_uav = self.common.get_uav_pc_pb(
                    self.usvs,
                    uav_idxs,
                    self.uavs[k],
                    k,
                    self.base,
                    self.usv_uav_channel_snapshot,
                )
                pc += pc_uav
                pb += pb_uav

        for i, usv in enumerate(self.usvs):
            decision = self.offloading_decisions[i]
            task_size_bits = usv['task_size']
            task_resource = usv['task_resource']

            # # todo
            # trans_time = 0.0
            # comp_time = 0.0
            # prop_delay = 0.0
            # allocated_resource = 0.0
            # allocated_rate = 0.0

            if decision == 0:  # Case 1: 本地计算
                comp_time = self.common.calculate_computation_time(task_size_bits, task_resource, usv['resource'])
                comp_energy = 0.1 * (1 ** 3) * comp_time   # point 1是以Ghz为单位的
                task_total_times[i] = comp_time
                usv_energies[i] = comp_energy
                if(comp_time <= self.base.task_completion_deadline):
                    completed_tasks += 1
                # print('usv ', i,"本地计算", ' comp time: ', comp_time, ' comp energy: ', comp_energy)

            elif 1 <= decision <= self.num_uavs:  # Case 2: 卸载到UAV
                uav_id = decision - 1
                uav = self.uavs[uav_id]

                # 通信过程
                channel_gain = self.usv_uav_channel_snapshot['gain'][i, uav_id]
                kappa = self.usv_uav_channel_snapshot['kappa'][i, uav_id]
                rate = self.common.calculate_rate_bps(
                    usv['power'],
                    channel_gain,
                    self.base.uav_bandwith,
                    kappa,
                )
                allocated_rate = rate * pb[i]  # point 使用分配的带宽资源
                trans_time = self.common.calculate_transmission_time(task_size_bits, allocated_rate)
                trans_energy = self.common.calculate_transmission_energy(usv['power'], trans_time)

                # 计算过程
                allocated_resource = uav['resource'] * pc[i]  # point 使用分配的计算资源
                comp_time = self.common.calculate_computation_time(task_size_bits, task_resource, allocated_resource)
                comp_energy = self.common.calculate_computation_energy(task_size_bits, task_resource,
                                                                       allocated_resource)

                task_total_times[i] = trans_time + comp_time
                usv_energies[i] = trans_energy
                uav_comp_energies[uav_id] += comp_energy
                # print("trans_energy: ", trans_energy, " comp_energy: ", comp_energy)
                if (trans_time + comp_time <= self.base.task_completion_deadline):
                    completed_tasks += 1
                # print('usv ', i, ' 卸载到UAV ', uav_id, ' comp time: ', comp_time, ' comp energy: ', comp_energy)

            else:  # Case 3: 卸载到卫星
                # 通信过程
                channel_gain = self.common.calculate_usv_to_satellite_channel_power_gain(usv['position'])
                rate = self.common.calculate_rate_bps(usv['power'], channel_gain, self.leo['bandwith'])
                allocated_rate = rate * pb[i]  # point 使用分配的带宽资源
                trans_time = self.common.calculate_transmission_time(task_size_bits, allocated_rate)
                trans_energy = self.common.calculate_transmission_energy(usv['power'], trans_time)
                prop_delay = self.base.H_LEO / self.base.light_speed  # 传播延迟

                # 计算过程 (使用分配的计算资源)
                allocated_resource = self.leo['resource'] * pc[i]  # 使用分配的计算资源
                comp_time = self.common.calculate_computation_time(task_size_bits, task_resource, allocated_resource)

                task_total_times[i] = trans_time + prop_delay + comp_time
                usv_energies[i] = trans_energy
                if (trans_time + comp_time <= self.base.task_completion_deadline):
                    completed_tasks += 1
                # print('usv ', i, ' 卸载到卫星 ', ' comp time: ', comp_time)


        # --- 3. 聚合系统级指标并计算奖励 ---
        self.system_time = np.sum(task_total_times) if self.num_usvs > 0 else 0


        total_usv_energy = np.sum(usv_energies)
        total_uav_energy = np.sum(uav_comp_energies) + np.sum(uav_fly_energies)
        self.total_energy = total_usv_energy + total_uav_energy

        self.avg_usv_energy = total_usv_energy / self.num_usvs if self.num_usvs > 0 else print('no usv')
        self.avg_uav_energy = total_uav_energy / self.num_uavs if self.num_uavs > 0 else print('no uav')

        # !! todo 新增：分别计算UAV的平均飞行和计算能耗 !!
        avg_uav_fly_energy = np.mean(uav_fly_energies) if self.num_uavs > 0 else 0
        avg_uav_comp_energy = np.mean(uav_comp_energies) if self.num_uavs > 0 else 0

        self.task_completion_rate = (completed_tasks / self.num_usvs) * 100 if self.num_usvs > 0 else 0
        # print('usv_completion_rate: ', self.task_completion_rate)

        # 奖励函数 (目标: 最小化时间)
        # !! 调用您设计的新的奖励函数 !!
        rewards_array = self.calculate_rewards(task_total_times, self.task_completion_rate, uav_comp_energies, uav_fly_energies)
        rewards = rewards_array.reshape(-1, 1)  # 转换为 (num_uavs, 1) 的形状

        # --- 4. 更新环境状态以备下一时间步 ---
        self.common.get_usv_mobility(self.usvs)   #point 这个函数内部已经实现了USV轨迹的记录

        # 为每个USV生成新任务 (在实际应用中，任务可能是随机到达的)
        for usv in self.usvs:
            usv['task_size'] = np.random.randint(self.base.task_size_min, self.base.task_size_max)
            usv['task_resource'] = np.random.randint(self.base.task_resources_min, self.base.task_resources_max)

        # print(f"step次数: {self.current_step}, 卸载决策: {self.offloading_decisions}")


        next_observation = self._get_observation()

        # --- 5. 返回结果 ---
        self.current_step += 1

        # 检查是否达到episode最大长度来判断是否结束
        is_done = self.current_step >= 60
        dones = [is_done] * self.num_uavs

        # info字典用于记录和调试，返回所有计算出的指标
        info = {
            "system_time": self.system_time,
            "total_energy": self.total_energy,
            "avg_usv_energy": self.avg_usv_energy,
            "avg_uav_energy": self.avg_uav_energy,
            "completion_rate": self.task_completion_rate,
            "avg_uav_fly_energy": avg_uav_fly_energy,
            "avg_uav_comp_energy": avg_uav_comp_energy,
        }

        # !! 关键改动：在回合结束时，将轨迹数据放入info !!
        if all(dones):
            info['trajectories'] = {
                'uavs': [uav['trajectory'] for uav in self.uavs],
                'usvs': [usv['trajectory'] for usv in self.usvs]
            }

        # if self.current_step % 5 == 0:  # todo 调试： 每5步记录一次
        #     for k in range(self.num_uavs):
        #         covered = self._count_nearby_usvs(k)
        #         velocity = np.linalg.norm(velocity_vector)
        #         print(f"Step {self.current_step}, UAV {k}: Covered USVs={covered}, "
        #               f"Fly Energy={uav_fly_energies[k]:.2f}, Speed={velocity:.2f}, "
        #               f"Completion Rate={self.task_completion_rate:.1f}%")

        return next_observation, rewards, dones, info
