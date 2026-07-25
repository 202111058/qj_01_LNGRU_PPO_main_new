import numpy as np
# from envs_202407.base import base
from .common import common

c1 = 10  # LoS/NLoS状态切换参数
c2 = 0.6  # LoS/NLoS状态切换参数
eta_los = 1.0  # LoS（视线）链路额外损耗，单位dB
eta_nlos = 20  # NLoS（非视线）链路额外损耗，单位dB
f_s = 2 * 10**9  # 载波频率，单位Hz
c = 3 * 10**8  # 光速
varpi0 = 1.58 * 10**(-13)  # 接收机噪声功率谱密度，通常单位W/Hz

def PotentialGame(usv, uav, leo, base, A_old):

    # 获取UD和SUAV的数量
    num_usv = base.num_usv  # todo M->num_usvs
    num_uav = base.num_uav  # todo N->num_uavs
    common_utils = common(base)

    # 初始化卸载决策向量A
    A = np.zeros(num_usv, dtype=int)
    A_temp = np.zeros(num_usv, dtype=int)
    A[0] = -1  # 确保第一次比较不相等

    # SECTION 先确定usv到所有uav的距离，如果小于coverage那么将该uav加入该usv可卸载的列表中
    d = np.zeros((num_usv, num_uav))
    for i in range(num_usv):
        for j in range(num_uav):
            d[i][j] = np.linalg.norm(usv[i]['position'] - uav[j]['position'])

    # print("有多少usv在uav的范围内d",d)

    # 遍历d，找出小于coverage的uav，新建一个矩阵表示二者关系，如果小于coverage，里面的值为1
    offload_opt = np.zeros((num_usv, num_uav))  # 初始化可用uav矩阵
    for i in range(num_usv):
        for j in range(num_uav):
            if d[i][j] < base.uav_coverage:
                offload_opt[i][j] = 1
    # print("有多少usv在uav的范围内offload_opt",offload_opt)

    # SECTION 如果有无人机在UAV的范围内，那么进入循环
    # 迭代直到卸载决策向量A不再改变
    # print("进入循环")
    while not np.array_equal(A, A_temp):
        # print(f"没被赋值的A{A},还有")
        A = A_temp.copy()
        # 创建本轮新决策的临时存储

        # SECTION 遍历所有USV，更新该USV的卸载决策（todo 如果卸载到UAV得是offload_opt里面的）
        for m in range(num_usv):
            # 初始化计算和带宽资源分配系数矩阵
            Pc_temp = np.zeros((num_usv, num_uav + 1))  # TODO 倒了一下，usv数量在前面，uav数量在后面，为什么只是加1，因为在本地不用算这个
            Pb_temp = np.zeros((num_usv, num_uav + 1))

            # # 重置当前USV的临时决策
            # A_temp[m] = 0

            # 初始化效用
            utility_temp = np.zeros(num_uav + 2)  # 0本地，1-num_uav 无人机  num_uav+1 卫星

            # SECTION 遍历所有USV，计算卸载到UAV或LEO时的Pc和Pb 也就是按照目前策略简档的所有决策的效用【给后面求和用！】
            for k in range(num_usv):  # note 都是遍历usv吗？
                selected_server_id = int(A_temp[k])  # 当前USV的卸载决策

                # SECTION 选中LEO
                if selected_server_id == num_uav + 1:  # note id=num_uav + 1表示LEO计算，但是存放位置是num_uav+1    # 当前卸载决策
                    # 提取UD参数
                    # gamma_T = PG_UDSet[k]['gamma_T']  # 不需要权重
                    # gamma_E = PG_UDSet[k]['gamma_E']
                    # P = usv[k]['power']  # 传输功率，单位w  能量计算才需要
                    D = usv[k]['task_size']  # 任务大小
                    eta = usv[k]['task_resource']  # 任务复杂度

                    # point 使用common类计算信道增益和传输速率
                    gain = common_utils.calculate_usv_to_satellite_channel_power_gain(usv[k]['position'])
                    R = common_utils.calculate_rate_bps(usv[k]['power'], gain, leo['bandwith'])


                    # 计算Pc和Pb
                    Pc_temp[k, num_uav] = np.sqrt(eta * D / leo['resource'])  # note 这算的都是分子  # 没有本地不用+1
                    Pb_temp[k, num_uav] = np.sqrt(D / R)

                # SECTION 选中UAV
                elif selected_server_id > 0 and selected_server_id <= num_uav :  # NOTE 这里id是1-num_uav
                    # 提取SUAV信息
                    selected_UAV = uav[int(selected_server_id) - 1]  # note 不一定需要-1
                    # 提取UD参数
                    # gamma_T = PG_UDSet[k]['gamma_T']
                    # gamma_E = PG_UDSet[k]['gamma_E']
                    # P = usv[k]['power']
                    D = usv[k]['task_size']
                    eta = usv[k]['task_resource']

                    # point 使用common类计算信道增益和传输速率
                    gain = common_utils.calculate_usv_to_uav_channel_power_gain(usv[k]['position'],
                                                                                      selected_UAV['position'])
                    R = common_utils.calculate_rate_bps(usv[k]['power'], gain, selected_UAV['bandwith'])

                    # 计算Pc和Pb
                    Pc_temp[k, int(selected_server_id) - 1] = np.sqrt(eta * D / selected_UAV['resource']) # note 这要倒一下,看一下gamma
                    Pb_temp[k, int(selected_server_id) - 1] = np.sqrt(D / R)

            # SECTION if当前本地处理效用
            eta_usv = usv[m]['task_resource']
            D_usv = usv[m]['task_size']
            T_loc = eta_usv * D_usv / usv[m]['resource']   # f应该是计算资源
            utility_temp[0] = T_loc  # note 为什么是1 要计算所有设备上的效用
            # print(f"usv{m}在本地计算的话用时为：{T_loc}")

            # note UD的最大允许延迟去掉了
            # SECTION if卫星计算效用
            # SNR_m = get_USV_to_LEO_SNR(usv[m], leo, base)  # 信噪比
            # r_m = leo['bandwith'] * np.log2(1 + SNR_m)  # 速率
            # point 使用common类计算信道增益和传输速率
            gain_m = common_utils.calculate_usv_to_satellite_channel_power_gain(usv[m]['position'])
            r_m = common_utils.calculate_rate_bps(usv[m]['power'], gain_m, leo['bandwith'])

            Pc_temp[m, num_uav] = np.sqrt(eta_usv * D_usv / leo['resource'])
            Pb_temp[m, num_uav] = np.sqrt(D_usv / r_m)
            f_leo = leo['resource'] * Pc_temp[m, num_uav] / np.sum(Pc_temp[:, num_uav]) # todo
            R_m = r_m * Pb_temp[m, num_uav] / np.sum(Pb_temp[:, num_uav])  # NOTE 乘了系数 # TODO
            T_leo_m = D_usv / R_m + eta_usv * D_usv / f_leo

            # print(f"在卫星计算的带宽系数行：{Pb_temp[:, num_uav]}")
            # print(f"usv{m}在卫星计算的话用时为：{T_leo_m}")
            # print(f"usv{m}在卫星计算的传输时延：{D_usv / R_m}")
            # print(f"usv{m}在卫星计算的计算时延：{eta_usv * D_usv / f_leo}")
            utility_temp[num_uav + 1] = T_leo_m  # NOTe 包含本地，需要+1，不是+2，因为num_uav就已经是个整数，无0

            # SECTION UAV计算效用 todo 得在有效的里面选，如果全无效就上面二选一
            for n in range(num_uav):
                if offload_opt[m][n] == 1:
                    # point 使用common类计算信道增益和传输速率
                    gain_m = common_utils.calculate_usv_to_uav_channel_power_gain(usv[m]['position'],
                                                                                  uav[n]['position'])
                    r_m = common_utils.calculate_rate_bps(usv[m]['power'], gain_m, uav[n]['bandwith'])

                    Pc_temp[m, n] = np.sqrt(eta_usv * D_usv / uav[n]['resource'])
                    Pb_temp[m, n] = np.sqrt(D_usv / r_m)
                    f_m = uav[n]['resource'] * Pc_temp[m, n] / np.sum(Pc_temp[:, n])
                    R_m = r_m * Pb_temp[m, n] / np.sum(Pb_temp[:, n])
                    T_uav_m = D_usv / R_m + eta_usv * D_usv / f_m
                    utility_temp[n + 1] = T_uav_m
                else:
                    utility_temp[n + 1] = np.inf

            # print(f"usv{m}的utility_temp {utility_temp}")
            # SECTION 找到除了0之外（排除掉不在范围内但是去找无人机的可能性）的最小成本，对应的UAV或LEO或本地处理
            index_temp = np.argmin(utility_temp)
            A_temp[m] = int(index_temp)


            # print(f"m{m}的utility_temp {utility_temp} ")
    # print("出循环")
    for i in range(num_usv):
        usv[i]['offload_decision'] = A[i]
    # note 返回卸载的无人机编号以及对应的pc，pb
    # print("usvs的卸载决策", A)
    return A

