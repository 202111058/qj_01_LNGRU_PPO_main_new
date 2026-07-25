# base.py
# 作用：定义所有仿真环境中的基础、静态参数。

import numpy as np

class base(object):
     def __init__(self):
        # =====================================================================
        # 1. 智能体与设备数量
        # =====================================================================
        self.num_uav = 4
        self.num_usv = 40
        self.leo_num = 1

        # =====================================================================
        # 2. 仿真场景参数
        # =====================================================================
        self.field_X = [0, 1000]  # 场景X轴范围 (米)
        self.field_Y = [0, 1000]  # 场景Y轴范围 (米)
        self.run_slot = 1.0  # 每个时间步的持续时间 (秒) - 已更新为1秒

        # =====================================================================
        # 3. 单位定义
        # =====================================================================
        self.Hz = 1
        self.kHz = 1000 * self.Hz
        self.MHz = 1000 * self.kHz
        self.GHz = 1000 * self.MHz
        self.bit = 1
        self.B = 8 * self.bit
        self.KB = 1024 * self.B
        self.MB = 1024 * self.KB

        # =====================================================================
        # 4. 任务参数
        # =====================================================================
        self.task_size_max = 1.0 * self.MB
        self.task_size_min = 0.5 * self.MB
        self.task_resources_max = 500  # 执行每bit任务所需的CPU周期数上限
        self.task_resources_min = 100  # 执行每bit任务所需的CPU周期数下限
        self.task_completion_deadline = 1.0 # 任务完成的判定时间 (秒)

        # =====================================================================
        # 5. USV参数 (海面)
        # =====================================================================
        self.usv_power = 0.2  # USV发射功率 (瓦特)
        self.usv_resource = 1 * self.GHz  # USV本地计算能力 (每秒周期数)，从0.3更新为1

        # =====================================================================
        # 6. UAV参数 (空中)
        # =====================================================================
        self.H_UAV = 100.0  # 无人机的飞行高度 (米)
        self.uav_resource = 25 * self.GHz  # UAV计算能力 (每秒周期数)
        self.uav_bandwith = 20 * self.MHz  # UAV负责的信道总带宽
        self.uav_coverage = 200  # 无人机覆盖范围 (米)
        self.tip_speed_rotor_blade = 120 # UAV旋翼叶尖速度 (米/秒)
        self.uav_energy_par1 = 80
        self.uav_energy_par2 = 22
        self.uav_energy_par3 = 263.4
        self.uav_energy_par4 = 0.0092

        # =====================================================================
        # 7. 卫星参数
        # =====================================================================
        self.H_LEO = 500 * 1000.0  # LEO卫星的高度 (米)，改为500公里
        self.leo_resource = 60 * self.GHz  # LEO卫星计算能力 (每秒周期数)
        self.leo_bandwith = 1 * self.GHz  # LEO卫星总带宽，改为1GHz
        self.light_speed = 3e8 # 光速 (米/秒)

        # =====================================================================
        # 8. 通信模型参数 (基于文献)
        # =====================================================================
        # 8.1 天线增益 (用户要求设为1，即0 dBi)
        self.usv_antenna_gain_db = 0.0
        self.uav_antenna_gain_db = 0.0
        self.leo_antenna_gain_db = 0.0

        # 8.2 USV-to-UAV 概率性视距路径损耗模型参数
        # 来源: "Joint Computation Offloading and Resource Allocation for Uncertain Maritime MEC..."
        # 该文引用了 "Hybrid Satellite-UAV-Terrestrial Networks..."
        self.zeta_L = 2.3  # LoS条件下的额外损耗 (dB)
        self.zeta_NL = 34   # NLoS条件下的额外损耗 (dB)
        self.alpha_uav_path_loss = 5.0188  # 环境参数a
        self.beta_uav_path_loss = 0.3511   # 环境参数b
        self.carrier_frequency_uav = 2 * self.GHz  # USV到UAV的载波频率

        # 8.3 USV-to-Satellite 路径损耗模型参数
        # 来源: "Performance Analysis of End-to-End LEO Satellite-Aided..."
        # 这里我们只使用大尺度衰落部分
        self.path_loss_exponent_sat = 2.4 # 路径损耗指数
        self.carrier_frequency_sat = 20 * self.GHz # 卫星通信常用Ka波段
        #——————小尺度衰落——————————————
        self.rician_K_factor_sat = 10

        # 8.4 噪声功率谱密度
        self.noise_power_density = 1.4e-13

        # =====================================================================
        # 9. 能耗模型参数
        # =====================================================================
        # 计算能耗系数, e.g., 10^-27
        self.energy_kappa = 1e-27

        # # 10. 奖励函数参数 (新增部分)
        # =====================================================================
        self.w_delay = 2         # 时延惩罚权重
        self.w_energy = 0.1        # 能耗惩罚权重
        self.w_boundary = 1      # 边界惩罚权重
        self.w_collision = 0.1     # 碰撞惩罚权重
        self.w_completion = 0.5    # 任务完成率奖励权重

        self.completion_threshold = 85.0  # 90%的任务完成率奖励阈值
        self.energy_threshold = 600.0   # 单个UAV的能耗惩罚阈值 (J)
        self.boundary_margin = 50.0      # 边界安全距离 (m)
        self.collision_threshold = 10.0  # UAV之间的安全距离 (m)   //仿照qwh论文
