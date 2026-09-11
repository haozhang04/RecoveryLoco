<p align="center">
  <img src="assets/image.png" alt="RecoveryLoco：Recovery and Locomotion" width="100%">
</p>

| Isaac Gym | MuJoCo | Physical |
|--- | --- | --- |
| ![isaacgym eval](assets/issacgym.gif) | ![mujoco eval](assets/mujoco.gif) | ![real eval](assets/real.gif) |

<div align="center">
  <h1 align="center">RecoveryLoco</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

<p align="center">
  <a href="https://www.bilibili.com/video/BV1Ln3i6RE3c">
    <img src="https://img.shields.io/badge/Bilibili-实机视频-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white" alt="Bilibili 实机视频">
  </a>
</p>

<p align="center">
  <strong>一种融合跌倒恢复与鲁棒运动训练的四足机器人盲走框架</strong>
</p>

---

# 🛠️ 1 安装

安装环境：
  - Ubuntu 22.04.5 LTS
  - NVIDIA Driver: 570.169 / CUDA: 12.8
  - Python: 3.8.20
  - PyTorch: 2.4.1+cu121
  - Isaac Gym: Preview 4

## 1.1 创建并激活 Conda 环境

```bash
conda create -n rl python=3.8.20
conda activate rl
```

## 1.2 安装 Isaac Gym

下载并解压 [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym)，进入其 Python 目录并安装：

```bash
cd isaacgym/python && pip install -e .
```

## 1.3 安装 Python 依赖

返回仓库根目录，安装 `requirements.txt` 中列出的第三方 Python 包：

```bash
python -m pip install -r requirements.txt
```

## 1.4 安装本地包

以 editable 模式安装仓库内的 `rsl_rl` 和 `legged_gym`：

```bash
./bash/leggedskill.sh -i
```

---

# 🚀 2 训练与运行

## 2.1 查看任务

列出所有已注册任务：

```bash
CUDA_VISIBLE_DEVICES=0 ./bash/leggedskill.sh -l
```

## 2.2 Train

训练 GO1 多 Critic 恢复运动策略：

```bash
CUDA_VISIBLE_DEVICES=0 ./bash/leggedskill.sh -t --task go1_recover_multi_critic
```

## 2.3 Play

### 2.3.1 最新策略

播放并导出最新策略：

```bash
CUDA_VISIBLE_DEVICES=0 ./bash/leggedskill.sh -p --task go1_recover_multi_critic
```

### 2.3.2 指定 checkpoint

通过 `--resume_path` 播放并导出指定 checkpoint，策略保存至对应任务目录的 `exported/policies/`：

```bash
CUDA_VISIBLE_DEVICES=0 ./bash/leggedskill.sh -p --task go1_recover_multi_critic --resume_path /path/to/model_1000.pt
```

## 2.4 Deploy

S2S（MuJoCo、Gazebo）和 S2R（Unitree Go1 实机）的部署代码及说明，
请参阅 [haozhang04/LeggedSkillDeploy](https://github.com/haozhang04/LeggedSkillDeploy) 仓库的 `main` 分支：

```bash
git clone https://github.com/haozhang04/LeggedSkillDeploy.git
```

---

# 📊 3 工具

## 3.1 日志压缩

预览实验日志压缩计划，添加 `--y` 后执行压缩：

```bash
python tools/logs_compress.py
```

## 3.2 旧模型清理

预览待删除权重，每个运行目录仅保留迭代次数最大的模型，添加 `--y` 后执行删除：

```bash
python tools/logs_delete.py
```

## 3.3 训练曲线

对比一个或多个 TensorBoard 日志目录的平均奖励和地形等级，图例使用目录名。
PDF 默认保存至最后一个输入目录的 `output/plot_reward_terrain_curves/`，也可通过 `--output_dir` 指定输出目录：

```bash
python tools/plot_reward_terrain_curves.py \
    --sample_step 10 \
    --input_dir logs/go1_recover_multi_critic/RUN_A/ logs/go1_recover_multi_critic/RUN_B/
```

## 3.4 地形恢复视频

离屏录制 GO1 从仰卧姿态恢复并通过指定地形，支持 `smooth_slope`、`rough_slope`、`stairs_up`、`stairs_down` 和 `obstacles`。
MP4 保存至运行目录的 `output/record_stairs_recovery_video/`：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/record_stairs_recovery_video.py \
    --terrain stairs_up \
    --input_dir logs/go1_recover_multi_critic/RUN/model_1000.pt
```

## 3.5 批量训练

在指定 GPU 上运行 8 个对比与消融任务，每张 GPU 最多并行运行两个任务，失败任务自动重试一次。
训练结果默认保存至 `logs/wo-TIMESTAMP/`，其中训练数据位于 `train_data/`，进程日志位于 `training_logs/TIMESTAMP/`：

```bash
python tools/train_tasks.py --gpu_ids 0 1
```

## 3.6 批量训练曲线

从批量训练日志生成全量、基线和消融对比曲线，共 6 个 PDF。
保存至所选日志批次的 `output/train_tasks_plot_reward_terrain_curves/`：

```bash
python tools/train_tasks_plot_reward_terrain_curves.py \
    --sample_step 10 \
    --input_dir logs/wo-TIMESTAMP/
```

---

# 📝 4 任务介绍

<table>
  <tr>
    <th>机器人</th>
    <th>任务</th>
    <th>说明</th>
  </tr>
  <tr>
    <td rowspan="6">GO1</td>
    <td><code>go1</code></td>
    <td>基础恢复运动</td>
  </tr>
  <tr>
    <td><code>go1_recover_bound</code></td>
    <td>融合跌倒恢复的 Bound 步态</td>
  </tr>
  <tr>
    <td><code>go1_recover_hop</code></td>
    <td>融合跌倒恢复的 Hop 步态</td>
  </tr>
  <tr>
    <td><code>go1_recover_handstand</code></td>
    <td>融合跌倒恢复的 倒立</td>
  </tr>
  <tr>
    <td><code>go1_recover_leggedstand</code></td>
    <td>融合跌倒恢复的 正立</td>
  </tr>
  <tr>
    <td><code>go1_recover_multi_critic</code></td>
    <td>结合 GRU 状态估计与多 Critic 的恢复运动</td>
  </tr>
</table>

---

# 🙏 致谢

本项目基于以下开源项目开发，感谢原作者的贡献：

| 项目 | 仓库链接 |
|------|----------|
| **HIMLoco** | [![GitHub](https://img.shields.io/badge/GitHub-HIMLoco-blue?logo=github)](https://github.com/InternRobotics/HIMLoco) |
| **legged_gym** | [![GitHub](https://img.shields.io/badge/GitHub-legged__gym-blue?logo=github)](https://github.com/leggedrobotics/legged_gym) |
| **rsl_rl** | [![GitHub](https://img.shields.io/badge/GitHub-rsl__rl-blue?logo=github)](https://github.com/leggedrobotics/rsl_rl.git) |

**如果本项目对你有帮助，欢迎点亮 Star ⭐！**
