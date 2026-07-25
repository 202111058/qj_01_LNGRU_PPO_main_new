# golden_point_test.py

import numpy as np
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# 步骤 1: 检查文件结构并从您的项目中导入真实的模块
# ---------------------------------------------------------------------------
try:
    import envs.envs_202509 as user_env

    print("成功导入您的环境文件...")
except (ImportError, ModuleNotFoundError) as e:
    print("=" * 80)
    print("错误：无法导入您的环境文件。请仔细检查您的文件结构是否如下所示：")
    print("""
    /你的项目文件夹/
    ├── golden_point_test.py  <-- 当前脚本
    └── /envs/
        ├── __init__.py         <-- 必须存在 (可以是空文件)
        ├── base.py
        ├── common.py
        ├── Init_player.py
        ├── Opi_RA_TO.py
        └── envs_202509.py
    """)
    print(f"详细错误信息: {e}")
    print("=" * 80)
    sys.exit(1)


def run_golden_point_test():
    """
    执行“黄金点”测试的核心函数。
    """
    print("=" * 20 + " “黄金点”测试开始 (使用真实环境文件) " + "=" * 20)

    env = user_env.EnvCore()
    print("环境 EnvCore 实例化成功...")

    env.reset()
    print(f"环境已重置，根据您的 Init_player.py 生成了 {len(env.usvs)} 个 USV。")

    if not env.usvs:
        print("错误：环境中没有USV，测试无法继续。")
        return

    usv_positions = np.array([usv['position'] for usv in env.usvs])
    num_clusters = env.num_uavs

    print(f"正在使用 K-Means 算法寻找 {num_clusters} 个最佳UAV部署中心点...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(usv_positions)

    golden_points = kmeans.cluster_centers_
    usv_labels = kmeans.labels_

    print("K-Means 聚类完成，找到的最佳中心点如下:")
    for i, point in enumerate(golden_points):
        print(f"  - UAV {i} 的黄金点: {point}")

    print("正在将每架UAV移动到其对应的黄金点...")
    for i, uav in enumerate(env.uavs):
        uav['position'] = golden_points[i].copy()

    stay_still_action = np.zeros(env.action_dim)
    actions = [stay_still_action for _ in range(env.num_uavs)]
    print("UAV将执行“保持静止”动作。")

    print("执行一步环境模拟...")
    _obs, _rewards, _dones, info = env.step(actions)
    _obs, _rewards, _dones, info = env.step(actions)

    print("\n" + "=" * 20 + " 测试结果分析 " + "=" * 20)
    print(f"在最理想的静态布局下，您的环境性能极限为：")
    decisions_for_analysis = info.get("decisions_this_step", "N/A")
    print(f"  - 卸载决策 (Offloading Decisions): {decisions_for_analysis}")
    print(f"  - 系统时延 (System Time): {info.get('system_time', 'N/A'):.3f} s")
    print(f"  - 任务完成率 (Completion Rate): {info.get('completion_rate', 'N/A'):.2f} %")
    print(f"  - 系统总能耗 (Total Energy): {info.get('total_energy', 'N/A'):.3f} J")
    print(f"  - 平均UAV能耗 (Avg. UAV Energy): {info.get('avg_uav_energy', 'N/A'):.3f} J")

    detailed_info = info.get("detailed_task_times", [])
    if detailed_info:
        print("\n--- 全任务详细分析 (Detailed Task Analysis) ---")
        for task_info in sorted(detailed_info, key=lambda x: x.get('total_time', 0), reverse=True):
            decision = task_info.get('decision', -1)
            task_id = task_info.get('task_id', 'N/A')

            if decision == 0:
                loc = "Local"
            elif decision > env.num_uavs:
                loc = "LEO"
            else:
                loc = f"UAV {decision - 1}"

            comp_res_ghz = task_info.get('allocated_comp_resource', 0) / 1e9
            comm_rate_mbps = task_info.get('allocated_comm_rate', 0) / 1e6

            time_breakdown = (f"Total={task_info.get('total_time', 0):.3f}s "
                              f"[Trans={task_info.get('trans_time', 0):.3f}, "
                              f"Comp={task_info.get('comp_time', 0):.3f}]")

            resource_breakdown = (f"Rate={comm_rate_mbps:.2f} Mbps, "
                                  f"CPU={comp_res_ghz:.2f} GHz")

            print(f"    Task {task_id:<2d} (to {loc:<7s}): {time_breakdown} | {resource_breakdown}")

    print("=" * 64)

    target_time = 20.0
    target_completion = 90.0
    system_time = info.get('system_time', float('inf'))
    completion_rate = info.get('completion_rate', 0)

    print("\n--- 诊断与建议 ---")
    if system_time > target_time or completion_rate < target_completion:
        print(f"结论：可能遇到了“环境瓶颈”。")
        print(f"即使在最理想的位置，系统性能（时延={system_time:.2f}s, 完成率={completion_rate:.2f}%）仍未达到您的目标。")
        print("建议：...")  # 省略部分建议打印
    else:
        print("结论：可能遇到了“策略瓶颈”。")
        print(f"好消息！您的环境本身有能力达到您的目标（测试时延为 {system_time:.2f}s, 完成率={completion_rate:.2f}%）。")
        print("建议：...")  # 省略部分建议打印

    print("\n正在生成最终布局的轨迹图...")
    plot_final_layout_new(env.usvs, env.uavs, golden_points, usv_labels)


# [重写后的绘图函数]
# [修改后的绘图函数 - 保存到文件]
def plot_final_layout_new(usvs, uavs, golden_points, usv_labels):
    """
    【新写法】绘制布局图并直接保存为文件，不进行屏幕显示。
    """
    # 1. 创建一个 Figure 和 Axes 对象
    fig, ax = plt.subplots(figsize=(10, 10))

    # (这部分绘图逻辑和之前完全一样)
    usv_positions = np.array([usv['position'] for usv in usvs])
    unique_labels = set(usv_labels)
    colors = [plt.cm.jet(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        class_member_mask = (usv_labels == k)
        xy = usv_positions[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=8, label=f'USV Cluster {k}')

    uav_positions = np.array([uav['position'] for uav in uavs])
    ax.scatter(uav_positions[:, 0], uav_positions[:, 1],
               s=250, c='red', marker='s', edgecolors='black',
               label='UAV Positions', zorder=3)

    ax.scatter(golden_points[:, 0], golden_points[:, 1],
               s=300, facecolors='none', edgecolors='yellow',
               linewidths=2, marker='*', label='Golden Points (K-Means Centers)',
               zorder=2)

    ax.set_title('Final Layout: USV and UAV Positions at Golden Points')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # --- [核心修改] ---
    # 2. 创建一个用于存放图片的文件夹
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)  # 如果文件夹已存在则什么也不做

    # 3. 定义保存路径和文件名
    save_path = os.path.join(plot_dir, "golden_point_layout.png")

    # 4. 保存图片到文件
    plt.savefig(save_path)

    # 5. 关闭图形，释放内存 (好习惯)
    plt.close(fig)

    # 6. 打印提示信息，告诉用户图片保存在哪里
    print(f"\n✅ 绘图已成功保存至: {save_path}")
    # --- [修改结束] ---


if __name__ == "__main__":
    run_golden_point_test()