import numpy as np
import math


class common:
    """
    封装所有可复用的核心计算函数。
    - 物理信道模型 (已根据最终要求重构)
    - 时间与能耗计算
    - 移动性模型
    """

    def __init__(self, base):
        self.base = base
        # 将dBm/Hz单位的噪声功率谱密度预先转换为W/Hz，便于后续计算
        # self.noise_power_density_W_Hz = 10 ** ((self.base.noise_power_density_dbm - 30) / 10)

    # ==============================================================================
    # 1. 通信模型 (Communication Models)
    # ==============================================================================

    def calculate_usv_to_uav_channel_power_gain(
        self, usv_pos, uav_pos, rician_power=1.0
    ):
        """
        计算USV到UAV的信道功率增益（线性值）。
        该模型是一个概率性视距（Probabilistic LoS）模型，能真实反映空对海信道特性。
        - 路径损耗模型来源: "Joint Computation Offloading and Resource Allocation..."
        - 环境参数来源: "Hybrid Satellite-UAV-Terrestrial Networks..."
        """
        dx = uav_pos[0] - usv_pos[0]
        dy = uav_pos[1] - usv_pos[1]
        dist_2d = np.sqrt(dx ** 2 + dy ** 2)
        if dist_2d < 1e-6: print("usv到uav的距离计算有问题")  # 避免除以零

        dist_3d = np.sqrt(dist_2d ** 2 + self.base.H_UAV ** 2)

        # 计算仰角 (弧度转角度)
        elevation_angle_deg = np.rad2deg(np.arctan(self.base.H_UAV / dist_2d))

        # LoS概率
        prob_los_exp_term = -self.base.beta_uav_path_loss * (elevation_angle_deg - self.base.alpha_uav_path_loss)
        prob_los = 1 / (1 + self.base.alpha_uav_path_loss * np.exp(prob_los_exp_term))

        # 自由空间路径损耗 (dB)
        fsl_db = 20 * np.log10(dist_3d) + 20 * np.log10(self.base.carrier_frequency_uav) - 147.55

        # 结合LoS和NLoS的最终路径损耗 (dB)
        path_loss_db = prob_los * (fsl_db + self.base.zeta_L) + (1 - prob_los) * (fsl_db + self.base.zeta_NL)

        # 将dB单位的路径损耗转换为线性的信道功率增益 (Gain = 1 / Loss)
        large_scale_gain = 10 ** (-path_loss_db / 10)

        # # 强制设为100% LoS来测试最好情况下的损耗
        # path_loss_db = fsl_db + self.base.zeta_L
        #
        # # 将dB单位的路径损耗转换为线性的信道功率增益 (Gain = 1 / Loss)
        # channel_power_gain = 10 ** (-path_loss_db / 10)

        return float(large_scale_gain * rician_power)

    def calculate_usv_to_uav_rician_power(self, distance_3d):
        """Return normalized Rician small-scale power for one access link."""
        K = self.base.rician_K_factor_uav
        xi = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2.0)
        phase = np.exp(
            -1j
            * 2.0
            * np.pi
            * distance_3d
            * self.base.carrier_frequency_uav
            / self.base.light_speed
        )
        h = (
            np.sqrt(K / (K + 1.0)) * phase
            + np.sqrt(1.0 / (K + 1.0)) * xi
        )
        return float(np.abs(h) ** 2)

    def calculate_usv_to_uav_doppler(self, usv, uav):
        """Return full Doppler, residual Doppler, and OFDM loss coefficient."""
        horizontal_delta = np.asarray(
            usv['position'], dtype=float
        ) - np.asarray(uav['position'], dtype=float)
        distance_3d = float(
            np.sqrt(
                np.dot(horizontal_delta, horizontal_delta)
                + self.base.H_UAV ** 2
            )
        )
        relative_velocity = (
            np.asarray(usv['velocity_vector'], dtype=float)
            - np.asarray(uav['velocity_vector'], dtype=float)
        )
        radial_velocity = float(
            np.dot(relative_velocity, horizontal_delta)
            / max(distance_3d, 1e-12)
        )
        doppler_hz = (
            radial_velocity
            * self.base.carrier_frequency_uav
            / self.base.light_speed
        )
        residual_hz = self.base.doppler_residual_ratio * doppler_hz
        kappa = float(
            np.sinc(
                residual_hz * self.base.ofdm_symbol_duration
            ) ** 2
        )
        return doppler_hz, residual_hz, float(np.clip(kappa, 0.0, 1.0))

    def build_usv_to_uav_channel_snapshot(self, usvs, uavs):
        """Sample every USV-UAV access link exactly once for the slot."""
        shape = (len(usvs), len(uavs))
        gains = np.zeros(shape, dtype=float)
        rician_powers = np.zeros(shape, dtype=float)
        doppler_hz = np.zeros(shape, dtype=float)
        residual_hz = np.zeros(shape, dtype=float)
        kappas = np.ones(shape, dtype=float)

        for k, usv in enumerate(usvs):
            for n, uav in enumerate(uavs):
                horizontal = float(
                    np.linalg.norm(usv['position'] - uav['position'])
                )
                distance_3d = float(np.hypot(horizontal, self.base.H_UAV))
                rician_power = self.calculate_usv_to_uav_rician_power(
                    distance_3d
                )
                fd, residual, kappa = self.calculate_usv_to_uav_doppler(
                    usv, uav
                )
                gains[k, n] = self.calculate_usv_to_uav_channel_power_gain(
                    usv['position'], uav['position'], rician_power
                )
                rician_powers[k, n] = rician_power
                doppler_hz[k, n] = fd
                residual_hz[k, n] = residual
                kappas[k, n] = kappa

        return {
            'gain': gains,
            'rician_power': rician_powers,
            'doppler_hz': doppler_hz,
            'residual_doppler_hz': residual_hz,
            'kappa': kappas,
        }

    def calculate_usv_to_satellite_channel_power_gain(self, usv_pos):
        """
        计算USV到卫星的信道功率增益（线性值）。
        模型包含大尺度路径损耗和小尺度莱斯衰落。
        - 大尺度衰落模型参考: "Performance Analysis of End-to-End..." , "Intelligent Task Scheduling..."
        - 小尺度衰落参数K因子参考: "Performance Analysis of End-to-End..."
        """
        # --- 1. 计算大尺度衰落 (路径损耗) ---
        dist = self.base.H_LEO  # 简化假设，距离约等于卫星高度

        # # 自由空间路径损耗 FSL (dB)
        # fsl_db = 20 * np.log10(dist) + 20 * np.log10(self.base.carrier_frequency_sat) - 147.55
        # 
        # # 考虑路径损耗指数的修正
        # large_scale_loss_db = fsl_db + 10 * (self.base.path_loss_exponent_sat - 2) * np.log10(dist)
        # 
        # # 转换为线性的大尺度功率增益
        # large_scale_gain_linear = 10 ** (-large_scale_loss_db / 10)

        L_fs = 20 * np.log10(4 * np.pi * dist * self.base.carrier_frequency_sat / self.base.light_speed)
        large_scale_gain_linear = 10 ** (-L_fs / 10)


        # --- 2. 计算小尺度衰落 (莱斯衰落) ---
        K = self.base.rician_K_factor_sat  # K = 10

        # LoS分量的确定性部分 (功率归一化)
        los_component = np.sqrt(K / (K + 1))

        # NLoS分量的随机部分 (功率归一化)
        # 均值为0，标准差为sqrt(1 / (2*(K+1))) 的复高斯随机数
        sigma = np.sqrt(1 / (2 * (K + 1)))
        nlos_component = (np.random.randn() + 1j * np.random.randn()) * sigma

        # 合成信道系数h
        h_rician = los_component + nlos_component

        # 小尺度功率增益 |h|^2
        small_scale_gain_linear = np.abs(h_rician) ** 2

        # --- 3. 合并总增益 ---
        # 总增益 = 大尺度增益 * 小尺度增益

        rx_gain_linear = 10 ** (30 / 10)

        total_channel_power_gain = large_scale_gain_linear * small_scale_gain_linear* rx_gain_linear
        return total_channel_power_gain

    def calculate_rate_bps(
        self,
        tx_power_w,
        channel_power_gain,
        bandwidth_hz,
        doppler_coefficient=1.0,
    ):
        """
        通用的香农公式计算速率 (返回bps)。
        这里的channel_power_gain已经包含了所有增益和损耗。
        天线增益在此版本中假定为1。
        """
        noise_power_w = self.base.noise_power_density


        # 接收功率 = 发射功率 * 信道功率增益 (已省略天线增益)
        rx_power_w = tx_power_w * channel_power_gain

        kappa = float(np.clip(doppler_coefficient, 0.0, 1.0))
        denominator = noise_power_w + rx_power_w * (1.0 - kappa)
        if denominator <= 0.0:
            return 0.0
        effective_sinr = rx_power_w * kappa / denominator
        return float(bandwidth_hz * np.log2(1.0 + effective_sinr))

    # ==============================================================================
    # 2. 时间与能耗计算 (Time & Energy Models)
    # ==============================================================================

    def calculate_transmission_time(self, task_size_bits, rate_bps):
        """计算传输时间 (秒)"""
        if rate_bps <= 1e-9: return float('inf')  # 防止除以零
        return task_size_bits / rate_bps

    def calculate_computation_time(self, task_size_bits, task_resource_cycles_per_bit, device_cpu_freq_cycles_per_sec):
        """计算计算时间 (秒)"""
        if device_cpu_freq_cycles_per_sec <= 0: return float('inf')
        total_cycles = task_size_bits * task_resource_cycles_per_bit
        return total_cycles / device_cpu_freq_cycles_per_sec

    def calculate_transmission_energy(self, tx_power_w, transmission_time_s):
        """计算传输能耗 (焦耳)"""
        return tx_power_w * transmission_time_s

    def calculate_computation_energy(self, task_size_bits, task_resource_cycles_per_bit,
                                     device_cpu_freq_cycles_per_sec):
        """计算计算能耗 (焦耳)"""
        en_uav_comp = 8.2e-27 * task_size_bits * task_resource_cycles_per_bit
        # print("""UAV计算能耗: {}焦耳""".format(en_uav_comp))
        return en_uav_comp

    def get_uav_fly_energy(self, velocity):
        """计算UAV在一个时间步内的飞行能耗 (焦耳)"""
        uav_travel_energy = (self.base.uav_energy_par1 * (
                1 + 3 * np.linalg.norm(velocity) ** 2 / self.base.tip_speed_rotor_blade ** 2)
                             + self.base.uav_energy_par2 * np.sqrt(
                    np.sqrt(self.base.uav_energy_par3 + np.linalg.norm(velocity) ** 4 / 4) - np.linalg.norm(
                        velocity) ** 2 / 2)
                             + self.base.uav_energy_par4 * np.linalg.norm(velocity) ** 3)
        # print("""UAV飞行能耗: {}焦耳""".format(uav_travel_energy))
        return uav_travel_energy

    # ==============================================================================
    # 3. 移动性模型 (Mobility Models)
    # ==============================================================================

    def get_usv_mobility(self, usv_list):
        """更新所有USV的位置"""
        # ... (此部分代码与上一版本相同，为简洁省略，您可以直接复制)
        alpha = 0.9
        avg_v = 1.0
        v_max = 8.0
        sigma_v = 2.0
        avg_d = np.random.rand() * 2 * np.pi
        sigma_d = 2

        for usv in usv_list:
            old_position = usv['position'].copy()
            rand_v = sigma_v * np.random.randn()
            rand_d = sigma_d * np.random.randn()
            usv['velocity'] = alpha * usv['velocity'] + (1 - alpha) * avg_v + np.sqrt(1 - alpha ** 2) * rand_v
            usv['velocity'] = np.clip(usv['velocity'], 0, v_max)
            usv['direction'] = alpha * usv['direction'] + (1 - alpha) * avg_d + np.sqrt(1 - alpha ** 2) * rand_d
            usv['direction'] %= (2 * np.pi)
            dx = usv['velocity'] * np.cos(usv['direction'])
            dy = usv['velocity'] * np.sin(usv['direction'])
            usv['position'][0] += dx
            usv['position'][1] += dy
            if not (self.base.field_X[0] < usv['position'][0] < self.base.field_X[1]):
                usv['direction'] = np.pi - usv['direction'] if self.base.field_Y[0] < usv['position'][1] < \
                                                               self.base.field_Y[1] else -usv['direction']
                usv['position'][0] = np.clip(usv['position'][0], self.base.field_X[0], self.base.field_X[1])
            if not (self.base.field_Y[0] < usv['position'][1] < self.base.field_Y[1]):
                usv['direction'] = -usv['direction']
                usv['position'][1] = np.clip(usv['position'][1], self.base.field_Y[0], self.base.field_Y[1])
            usv['velocity_vector'] = (
                usv['position'] - old_position
            ) / self.base.run_slot
            usv['trajectory'].append(np.copy(usv['position']))

    # 在common.py中添加以下函数

    def get_leo_pc_pb(self, usvs, idxs, leo, base):
        """
        计算卸载到卫星的USV的计算和带宽资源分配系数。

        参数:
        - usvs: USV列表
        - idxs: 卸载到卫星的USV索引列表
        - leo: 卫星对象
        - base: 基础参数对象

        返回:
        - pc: 计算资源分配系数向量
        - pb: 带宽资源分配系数向量
        """
        num_usvs = len(usvs)
        pc_temp = np.zeros(num_usvs)
        pb_temp = np.zeros(num_usvs)

        for k in idxs:
            D = usvs[k]['task_size']
            eta = usvs[k]['task_resource']

            # 计算传输速率
            gain = self.calculate_usv_to_satellite_channel_power_gain(usvs[k]['position'])
            r = self.calculate_rate_bps(usvs[k]['power'], gain, leo['bandwith'])

            # 计算资源分配系数
            pc_temp[k] = np.sqrt(eta * D / leo['resource'])
            pb_temp[k] = np.sqrt(D / r)

        # 归一化
        pc_sum = np.sum(pc_temp[idxs])
        pb_sum = np.sum(pb_temp[idxs])

        if pc_sum > 0:
            pc_temp[idxs] = pc_temp[idxs] / pc_sum
        if pb_sum > 0:
            pb_temp[idxs] = pb_temp[idxs] / pb_sum

        return pc_temp, pb_temp

    def get_uav_pc_pb(
        self, usvs, idxs, uav, uav_index, base, channel_snapshot
    ):
        """
        计算卸载到UAV的USV的计算和带宽资源分配系数。

        参数:
        - usvs: USV列表
        - idxs: 卸载到UAV的USV索引列表
        - uav: UAV对象
        - base: 基础参数对象

        返回:
        - pc: 计算资源分配系数向量
        - pb: 带宽资源分配系数向量
        """
        num_usvs = len(usvs)
        pc_temp = np.zeros(num_usvs)
        pb_temp = np.zeros(num_usvs)

        for k in idxs:
            D = usvs[k]['task_size']
            eta = usvs[k]['task_resource']

            # 计算传输速率
            gain = channel_snapshot['gain'][k, uav_index]
            kappa = channel_snapshot['kappa'][k, uav_index]
            r = self.calculate_rate_bps(
                usvs[k]['power'], gain, uav['bandwith'], kappa
            )

            # 计算资源分配系数
            pc_temp[k] = np.sqrt(eta * D / uav['resource'])
            pb_temp[k] = np.sqrt(D / r)

        # 归一化
        pc_sum = np.sum(pc_temp[idxs])
        pb_sum = np.sum(pb_temp[idxs])

        if pc_sum > 0:
            pc_temp[idxs] = pc_temp[idxs] / pc_sum
        if pb_sum > 0:
            pb_temp[idxs] = pb_temp[idxs] / pb_sum

        return pc_temp, pb_temp
