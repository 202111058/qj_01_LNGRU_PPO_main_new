# 多跳 UAV-BS 实验迁移与运行指南

> 适用项目：`lngru_mappo-main`  
> 适用实验：UAV 仅承担通信中继、USV 任务只选择本地计算或地面 BS 计算的多跳对比实验  
> 推荐环境：Python 3.10  
> 文档日期：2026 年 7 月 25 日

这份指南用于把当前已经实现的多跳 UAV-BS 实验迁移到另一台电脑。它按照“准备原项目—复制新增文件—安装环境—运行测试—开始训练—查看结果”的顺序编写，只要另一台电脑上的原项目版本一致，就可以逐个复制文件并按照命令完成部署。

如果不能确认另一台电脑上的原项目是否与当前版本完全一致，最稳妥的做法是复制整个 `lngru_mappo-main` 项目，然后排除缓存和已有训练结果。只复制四个新增文件的方式，前提是目标电脑已经具有同版本的原项目代码。

---

## 1. 迁移前提与最短流程

迁移时应当始终以 `lngru_mappo-main` 作为项目根目录。最短迁移流程如下：

1. 在另一台电脑准备同版本的原始 `lngru_mappo-main` 项目。
2. 把四个多跳运行文件放进对应目录。
3. 建议同时复制三个测试文件和本说明文档。
4. 创建 Python 3.10 Conda 环境并安装依赖。
5. 进入项目根目录，先执行导入检查。
6. 运行全部 18 项多跳自动测试。
7. 测试通过后先跑 6 万步预训练。
8. 预训练稳定后运行 30 万步正式训练。
9. 用 TensorBoard 查看奖励、时延、能耗、卸载率和跳数等结果。

迁移后的目标目录结构至少应当包含：

```text
lngru_mappo-main/
├── algorithms/
├── envs/
│   ├── base.py
│   ├── Init_player.py
│   ├── common.py
│   ├── env_wrappers.py
│   ├── envs_multihop_bs.py
│   └── env_continuous_multihop.py
├── runner/
│   └── shared/
│       ├── base_runner.py
│       ├── env_runner.py
│       └── env_runner_multihop.py
├── tests/
│   ├── __init__.py
│   ├── test_multihop_bs.py
│   └── test_multihop_entrypoint.py
├── train/
│   ├── train.py
│   └── train_multihop.py
├── utils/
└── config.py
```

---

## 2. 必须复制的四个运行文件

正式运行多跳实验必须复制下面四个文件。源文件和目标文件应当保持完全相同的项目相对路径。

| 序号 | 必须复制的文件 | 放置目录 | 作用 |
|---:|---|---|---|
| 1 | `envs/envs_multihop_bs.py` | `envs/` | 多跳核心环境，包含 BS、通信图、Dijkstra 路由、资源分配、二元势博弈、时延、能耗、奖励和环境状态转移 |
| 2 | `envs/env_continuous_multihop.py` | `envs/` | Gym 连续动作封装，连接多跳环境和现有 RMAPPO 训练框架 |
| 3 | `train/train_multihop.py` | `train/` | 独立的多跳训练入口，固定使用 `MultihopUAVBS` 结果命名空间 |
| 4 | `runner/shared/env_runner_multihop.py` | `runner/shared/` | 多跳训练循环和日志记录，保存 BS 卸载率、路径可用率、跳数、势博弈收敛和中继能耗等指标 |

复制后的目录位置必须是：

```text
lngru_mappo-main/envs/envs_multihop_bs.py
lngru_mappo-main/envs/env_continuous_multihop.py
lngru_mappo-main/train/train_multihop.py
lngru_mappo-main/runner/shared/env_runner_multihop.py
```

不要把四个文件全部放进项目根目录，也不要改变文件名，因为代码中的导入语句依赖上述目录结构。

本次多跳实验采用新增独立文件的方式，没有要求替换以下原实验文件：

```text
envs/envs_202509.py
envs/env_continuous.py
envs/Opi_RA_TO.py
envs/common.py
train/train.py
runner/shared/env_runner.py
config.py
```

因此，原实验仍然通过 `train/train.py` 启动，多跳实验只通过 `train/train_multihop.py` 启动。两组实验的训练入口和结果目录相互独立。

多跳环境会在 `MultihopBase` 中显式固定 `4` 架 UAV、`20` 艘 USV
和 `60` 个时隙，不继承原实验可能正在进行的 USV 数量扫描。这样才能保证
文档中的单智能体观测维度 `90` 和共享观测维度 `360` 保持不变。

---

## 3. 建议复制和不需要复制的文件

### 3.1 建议同时复制的测试文件

测试文件不参与正式训练，但强烈建议复制。它们可以验证文件是否放对位置、环境是否安装完整，以及多跳逻辑是否在迁移过程中被破坏。

```text
tests/__init__.py
tests/test_multihop_bs.py
tests/test_multihop_entrypoint.py
```

其中，`test_multihop_bs.py` 主要验证拓扑、路由、资源分配、势博弈、时延、能耗、奖励和环境步进；`test_multihop_entrypoint.py` 主要验证 Gym 封装、随机种子、60 时隙 episode、训练参数约束和日志入口。

### 3.2 建议复制的说明文件

下面的说明材料不参与训练，但便于在另一台电脑继续学习、汇报和排查问题：

```text
docs/多跳UAV-BS实验迁移与运行指南.md
docs/多跳UAV中继卸载策略与正式实验指南.docx
docs/superpowers/specs/2026-07-21-multihop-uav-bs-design.md
docs/superpowers/plans/2026-07-21-multihop-uav-bs-implementation.md
```

当前这份 Markdown 是部署操作手册；Word 文档重点解释多跳策略如何选择路径、如何使用势博弈决定本地或 BS，以及如何向老师和师姐汇报；设计和实现记录用于追溯参数与建模依据。

### 3.3 不需要复制的缓存和系统文件

以下内容可以不复制，它们会在新电脑运行时自动生成，或者只与当前电脑的编辑器和操作系统有关：

```text
__pycache__/
*.pyc
.DS_Store
.idea/
*.log
```

如果准备从头正式训练，也不需要复制：

```text
results/
```

`results` 中保存的是当前电脑已经生成的模型、TensorBoard 日志和轨迹图片，不属于运行代码。只有需要保留已有实验记录、查看当前冒烟测试结果或转移已有模型时，才需要单独复制这个目录。

---

## 4. 原项目依赖关系

四个新增运行文件不能放在一个空文件夹中独立运行，它们会复用原项目中的环境参数、USV/UAV 初始化、通信模型、向量环境、RMAPPO 网络、经验缓冲区和训练器。

主要依赖关系如下：

```text
train/train_multihop.py
├── config.py
├── envs/env_continuous_multihop.py
├── envs/env_wrappers.py
├── envs/envs_multihop_bs.py
└── runner/shared/env_runner_multihop.py

envs/envs_multihop_bs.py
├── envs/base.py
├── envs/Init_player.py
└── envs/common.py

runner/shared/env_runner_multihop.py
└── runner/shared/env_runner.py
    ├── runner/shared/base_runner.py
    ├── algorithms/
    └── utils/
```

因此，另一台电脑至少还必须具有下面这些原项目文件和目录：

```text
config.py
algorithms/
utils/
envs/base.py
envs/Init_player.py
envs/common.py
envs/env_wrappers.py
runner/shared/base_runner.py
runner/shared/env_runner.py
```

如果另一台电脑上已经有同版本的原项目，可以只复制新增文件进行覆盖式补充。如果无法确认原项目版本，建议复制整个项目目录，并排除第 3.3 节列出的缓存、编辑器文件和旧结果。

---

## 5. Python 与 Conda 环境安装

当前已经完成测试的环境版本如下：

| 软件包 | 已验证版本 |
|---|---:|
| Python | 3.10.6 |
| NumPy | 1.23.5 |
| Gym | 0.26.2 |
| PyTorch | 2.0.0 |
| SciPy | 1.9.3 |
| Matplotlib | 3.6.2 |
| TensorBoard | 2.16.2 |
| TensorBoardX | 2.6.2.2 |
| setproctitle | 1.3.3 |

### 5.1 创建 Conda 环境

打开终端或 Anaconda Prompt，执行：

```bash
conda create -n multihop python=3.10 -y
conda activate multihop
```

检查 Python 版本：

```bash
python --version
```

预期看到 Python 3.10.x。

### 5.2 安装通用依赖

执行：

```bash
pip install numpy==1.23.5 gym==0.26.2 scipy==1.9.3 matplotlib==3.6.2 tensorboard==2.16.2 tensorboardX==2.6.2.2 setproctitle==1.3.3
```

### 5.3 安装 PyTorch

如果另一台电脑只使用 CPU，可以安装当前已经验证的版本：

```bash
pip install torch==2.0.0
```

如果另一台电脑使用 NVIDIA GPU，不建议机械复制当前 Mac 的 PyTorch 安装方式。应当先确认该电脑的显卡驱动和 CUDA 版本，再安装与 CUDA 匹配的 PyTorch；安装完成后使用下面的命令检查：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
```

GPU 环境中，第二行应当输出：

```text
cuda available: True
```

### 5.4 检查主要依赖

```bash
python -c "import numpy, gym, torch, scipy, matplotlib, tensorboardX, setproctitle; print('dependencies: OK')"
```

预期输出：

```text
dependencies: OK
```

---

## 6. CPU、GPU 与 `--cuda` 参数

当前项目中的 `--cuda` 参数采用了容易误解的反向定义。配置代码使用的是 `action="store_false"`，因此其实际行为如下：

| 启动方式 | `all_args.cuda` | 实际设备选择 |
|---|---:|---|
| 命令中不写 `--cuda` | `True` | 如果 PyTorch 检测到 CUDA，则使用 GPU；否则回退到 CPU |
| 命令中写上 `--cuda` | `False` | 强制关闭 CUDA，使用 CPU |

因此：

- 在 Mac、无 NVIDIA 显卡的电脑或明确使用 CPU 时，训练命令末尾保留 `--cuda`。
- 在已经正确安装 CUDA 版 PyTorch 的 NVIDIA GPU 服务器上，如果希望使用 GPU，则不要写 `--cuda`。
- 判断是否真正使用 GPU，应同时观察训练开始时的终端输出。GPU 模式会输出 `choose to use gpu...`，CPU 模式会输出 `choose to use cpu...`。

这个参数名称虽然容易产生误解，但本次迁移不修改原来的参数定义，以免改变现有训练脚本的行为。

---

## 7. 文件复制后的导入检查与 18 项自动测试

所有命令都应当在项目根目录 `lngru_mappo-main` 下执行。先进入项目目录：

```bash
cd /你的实际路径/lngru_mappo-main
```

Windows 用户可以在 Anaconda Prompt 中写成：

```text
cd /d D:\你的实际路径\lngru_mappo-main
```

### 7.1 检查核心环境导入

```bash
python -c "from envs.envs_multihop_bs import EnvCore; env = EnvCore(); print('obs_dim:', env.obs_dim); print('action_dim:', env.action_dim)"
```

预期输出：

```text
obs_dim: 90
action_dim: 2
```

如果出现 `ModuleNotFoundError`，优先检查以下问题：

1. 当前终端是否位于项目根目录。
2. 四个文件是否放在正确的相对目录。
3. 文件名是否被聊天软件或操作系统自动增加了 `(1)`、`.txt` 等后缀。
4. Conda 环境是否已经激活。

### 7.2 检查 Gym 封装

```bash
python -c "from envs.env_continuous_multihop import ContinuousMultihopEnv; env = ContinuousMultihopEnv(); obs = env.reset(); print('observation shape:', obs.shape)"
```

预期输出：

```text
observation shape: (4, 90)
```

### 7.3 运行 18 项自动测试

```bash
python -m unittest discover -s tests -p "test_multihop*.py" -v
```

测试通过时，末尾应显示：

```text
----------------------------------------------------------------------
Ran 18 tests in ...s

OK
```

测试过程中可能出现 Matplotlib 字体缓存或 Gym 版本提示，只要最后是 `Ran 18 tests` 和 `OK`，就表示多跳代码逻辑、训练入口和日志接口均已通过测试。如果最后出现 `FAILED` 或 `ERROR`，不要直接开始 30 万步训练，应先根据第一条失败测试排查文件路径、依赖版本和代码是否完整。

---

## 8. 6 万步预训练与 30 万步正式训练

建议先运行 6 万步预训练，再运行 30 万步正式训练。这样可以在投入较长训练时间前发现环境、日志、数值或设备配置问题。

以下多行命令适用于 macOS/Linux Bash。Windows 用户可以把各行合并成一行执行，或者在 PowerShell 中使用反引号代替行末的反斜杠。

### 8.1 CPU：6 万步预训练

```bash
python train/train_multihop.py \
  --algorithm_name rmappo \
  --experiment_name multihop_pilot_seed1 \
  --seed 1 \
  --num_env_steps 60000 \
  --episode_length 60 \
  --n_rollout_threads 5 \
  --n_training_threads 2 \
  --cuda
```

CPU 模式下保留最后的 `--cuda`。训练开始时应当看到：

```text
choose to use cpu...
Starting multihop UAV-BS experiment ...
```

### 8.2 CPU：30 万步正式训练

```bash
python train/train_multihop.py \
  --algorithm_name rmappo \
  --experiment_name multihop_formal_seed1 \
  --seed 1 \
  --num_env_steps 300000 \
  --episode_length 60 \
  --n_rollout_threads 5 \
  --n_training_threads 2 \
  --cuda
```

论文正式结果建议至少运行三个随机种子。第二组和第三组分别修改为：

```text
--experiment_name multihop_formal_seed2 --seed 2
--experiment_name multihop_formal_seed3 --seed 3
```

每组实验都应保持相同的总步数、episode 长度、rollout 环境数量、训练线程数和评价规则，只改变实验名称与随机种子。

### 8.3 GPU 训练

GPU 训练参数与上面相同，但必须删除最后的 `--cuda`。例如 30 万步 GPU 正式训练：

```bash
python train/train_multihop.py \
  --algorithm_name rmappo \
  --experiment_name multihop_formal_seed1 \
  --seed 1 \
  --num_env_steps 300000 \
  --episode_length 60 \
  --n_rollout_threads 5 \
  --n_training_threads 2
```

训练开始时应当看到：

```text
choose to use gpu...
```

如果 GPU 服务器仍然输出 `choose to use cpu...`，说明 PyTorch 没有检测到 CUDA。此时应检查 `torch.cuda.is_available()`，而不是继续等待长时间训练。

---

## 9. TensorBoard 与结果保存位置

### 9.1 结果目录

多跳实验结果统一保存在：

```text
results/MultihopUAVBS/MyEnv/rmappo/<experiment_name>/runN/
```

例如，第一次运行 `multihop_formal_seed1` 时通常得到：

```text
results/MultihopUAVBS/MyEnv/rmappo/multihop_formal_seed1/run1/
├── models/
│   ├── actor.pt
│   └── critic.pt
├── train/
│   ├── events.out.tfevents...
│   └── summary.json
└── trajectories/
    ├── episode_0.png
    ├── episode_1.png
    └── ...
```

各目录含义如下：

| 目录或文件 | 含义 |
|---|---|
| `models/actor.pt` | 训练后的策略网络 |
| `models/critic.pt` | 训练后的价值网络 |
| `train/events.out.tfevents...` | TensorBoard 曲线的主要数据来源 |
| `train/summary.json` | TensorBoardX 导出的标量摘要；部分运行中可能为空 |
| `trajectories/` | 不同训练阶段的 UAV 轨迹图片 |

同一个 `experiment_name` 再次运行时，程序会自动创建 `run2`、`run3`，不会覆盖已有的 `run1`。不同随机种子仍建议使用不同的 `experiment_name`，方便后续统计。

### 9.2 启动 TensorBoard

在项目根目录执行：

```bash
tensorboard --logdir results/MultihopUAVBS --port 6006
```

然后在浏览器打开：

```text
http://localhost:6006
```

如果终端提示找不到 `tensorboard` 命令，可以使用：

```bash
python -m tensorboard.main --logdir results/MultihopUAVBS --port 6006
```

### 9.3 建议重点查看的指标

| 指标 | 含义与检查重点 |
|---|---|
| `episode_reward` | 每回合平均奖励，主要用于观察训练是否逐渐收敛 |
| `system_time` | 20 个 USV 任务的系统服务时延，通常越低越好 |
| `completion_rate` | 1 秒截止时间内完成的任务比例 |
| `total_energy` | USV 与 UAV 的总能耗 |
| `avg_uav_energy` | UAV 平均飞行和中继能耗 |
| `avg_usv_energy` | USV 平均本地计算或接入传输能耗 |
| `avg_uav_comp_energy` | UAV 任务计算能耗；本方案中应始终为 0 |
| `avg_uav_relay_energy` | UAV 多跳转发产生的平均中继能耗 |
| `route_availability_ratio` | 当前时隙具有完整 UAV-BS 路径的 USV 比例 |
| `bs_offloading_ratio` | 最终通过势博弈选择 BS 卸载的任务比例 |
| `avg_hop_count` | BS 卸载任务的平均无线跳数，包含 USV 接入跳 |
| `max_hop_count` | 当前回合出现的最大无线跳数 |
| `potential_passes` | 势博弈完成一轮卸载决策所需的遍历轮数 |
| `potential_converged` | 势博弈是否收敛，正常情况下应当为 1 |

训练是否正常不能只根据奖励单一判断。至少应同时检查 `potential_converged`、`avg_uav_comp_energy`、`route_availability_ratio`、`bs_offloading_ratio`、系统时延和完成率。

---

## 10. 最终迁移检查清单

在另一台电脑开始 30 万步正式训练前，逐项完成下面的检查。

### 文件与目录

- [ ] 另一台电脑已经有完整的同版本 `lngru_mappo-main` 原项目。
- [ ] `envs/envs_multihop_bs.py` 已放入 `envs/`。
- [ ] `envs/env_continuous_multihop.py` 已放入 `envs/`。
- [ ] `train/train_multihop.py` 已放入 `train/`。
- [ ] `runner/shared/env_runner_multihop.py` 已放入 `runner/shared/`。
- [ ] 三个测试文件已放入 `tests/`。
- [ ] 文件名没有被增加 `(1)`、`.txt` 或其他后缀。
- [ ] 没有把 `__pycache__`、`.DS_Store` 或 `.idea` 当作必要代码复制。

### Python 环境

- [ ] Conda 环境已经创建并激活。
- [ ] `python --version` 显示 Python 3.10.x。
- [ ] NumPy、Gym、PyTorch、SciPy、Matplotlib、TensorBoardX 和 setproctitle 可以正常导入。
- [ ] 如果使用 GPU，`torch.cuda.is_available()` 返回 `True`。

### 代码验证

- [ ] 核心环境导入结果为 `obs_dim: 90`、`action_dim: 2`。
- [ ] Gym 封装的观测形状为 `(4, 90)`。
- [ ] 自动测试末尾显示 `Ran 18 tests`。
- [ ] 自动测试最终状态为 `OK`。

### 训练与日志

- [ ] CPU 训练命令保留 `--cuda`，GPU 训练命令删除 `--cuda`。
- [ ] 训练终端显示的实际设备与预期一致。
- [ ] 6 万步预训练可以持续运行，没有 NaN、shape error 或势博弈异常。
- [ ] `potential_converged` 为 1，`avg_uav_comp_energy` 为 0。
- [ ] `results/MultihopUAVBS/` 下已经生成独立结果目录。
- [ ] TensorBoard 能够打开，并能看到奖励、时延、完成率、卸载率、跳数和能耗曲线。
- [ ] 正式训练使用 30 万步，并为不同随机种子设置不同的实验名称。

完成以上检查后，再开始 `seed=1、2、3` 的正式训练。这样能够最大限度避免因为文件漏传、环境版本不一致、CPU/GPU 参数写反或日志路径错误而浪费训练时间。
