# Init_player_v1.py
# 作用：初始化场景中的各个实体，如USV, UAV, LEO和任务。

import numpy as np

# 定义中央区域
CENTRAL_REGION = {"name": "中央", "x_range": (300, 700), "y_range": (300, 700)}

# --- 修改部分 1: 定义固定的角落坐标 ---
# 我们不再使用随机范围，而是定义一个包含精确[x, y]坐标的列表
FIXED_CORNER_COORDINATES = [
    [100.0, 100.0],  # 左下角
    [900.0, 100.0],  # 右下角
    [100.0, 900.0],  # 左上角
    [900.0, 900.0]  # 右上角
]
# ----------------------------------------

# (原来的CORNER_POSITIONS可以保留，也可以删除，因为它不再被使用)
CORNER_POSITIONS = [
    {"name": "左下", "x_range": (50, 150), "y_range": (50, 150)},
    {"name": "右下", "x_range": (850, 950), "y_range": (50, 150)},
    {"name": "左上", "x_range": (50, 150), "y_range": (850, 950)},
    {"name": "右上", "x_range": (850, 950), "y_range": (850, 950)}
]
# 定义核心用户比例，决定有多少用户严格生成在区域内
CORE_USER_RATIO = 0.8


def init_usv(base):
    usvs = []

    for i in range(base.num_usv):
        usv = {}
        # 在中央区域随机生成位置
        x_pos = np.random.uniform(CENTRAL_REGION["x_range"][0], CENTRAL_REGION["x_range"][1])
        y_pos = np.random.uniform(CENTRAL_REGION["y_range"][0], CENTRAL_REGION["y_range"][1])

        # 添加一些随机性，使分布更自然
        x_pos += np.random.normal(0, 10)
        y_pos += np.random.normal(0, 10)

        # 确保在场景范围内
        x_pos = np.clip(x_pos, base.field_X[0], base.field_X[1])
        y_pos = np.clip(y_pos, base.field_Y[0], base.field_Y[1])

        usv['position'] = np.array([x_pos, y_pos])
        usv['offload_decision'] = 1
        usv['resource'] = base.usv_resource
        usv['power'] = base.usv_power
        usv['task_size'] = np.random.randint(base.task_size_min, base.task_size_max)
        usv['task_resource'] = np.random.randint(base.task_resources_min, base.task_resources_max)
        usv['velocity'] = 1.0
        usv['direction'] = np.random.rand() * 2 * np.pi
        usv['velocity_vector'] = usv['velocity'] * np.array(
            [np.cos(usv['direction']), np.sin(usv['direction'])],
            dtype=float,
        )
        usv['trajectory'] = [np.copy(usv['position'])]
        usvs.append(usv)
    return usvs


def init_uav(base):
    uavs = []
    for uav_id in range(base.num_uav):
        uav = {}

        # --- 修改部分 2: 从固定坐标列表中分配位置 ---
        # 从我们新定义的固定坐标列表中，按顺序为每个UAV分配位置
        # 使用 % (取余) 操作符确保即使UAV数量超过4个，也能循环使用这些位置
        fixed_pos = FIXED_CORNER_COORDINATES[uav_id % len(FIXED_CORNER_COORDINATES)]

        # 直接设置UAV的位置
        uav['position'] = np.array(fixed_pos, dtype=float)
        # -------------------------------------------

        # (注释或删除掉原来的随机生成代码)
        # corner_region = CORNER_POSITIONS[uav_id % len(CORNER_POSITIONS)]
        # x_pos = np.random.uniform(corner_region["x_range"][0], corner_region["x_range"][1])
        # y_pos = np.random.uniform(corner_region["y_range"][0], corner_region["y_range"][1])
        # uav['position'] = np.array([x_pos, y_pos])

        uav['bandwith'] = base.uav_bandwith
        uav['resource'] = base.uav_resource
        uav['high'] = base.H_UAV
        uav['number'] = uav_id
        uav['velocity_vector'] = np.zeros(2, dtype=float)
        uav['trajectory'] = [np.copy(uav['position'])]
        uavs.append(uav)
    return uavs


def init_leo(base):
    leo = {}
    leo['resource'] = base.leo_resource
    leo['bandwith'] = base.leo_bandwith
    leo['high'] = base.H_LEO
    return leo


def init_tasks(base):
    all_tasks = []
    for i in range(base.num_usv):
        task = {
            'task_size': np.random.randint(base.task_size_min, base.task_size_max),
            'task_resource': np.random.randint(base.task_resources_min, base.task_resources_max)
        }
        all_tasks.append(task)
    return all_tasks
