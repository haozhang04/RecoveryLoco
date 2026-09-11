#!/usr/bin/env python3
"""
Plot Mean Reward vs. Learning Iteration.
"""
import os
import re
import isaacgym
import matplotlib
from pathlib import Path

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym import LEGGED_GYM_LOGS_DIR


# 如果没有DISPLAY环境变量，使用非交互式后端
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
else:
    # 尝试使用TkAgg，如果失败则降级到Agg
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Agg')

import matplotlib.pyplot as plt


def parse_log(file_path: Path):
    """返回三等长列表：iterations, rewards 和 terrain_levels。"""
    iter_pat = re.compile(r"Learning iteration (\d+)")
    reward_pat = re.compile(r"Mean reward:\s*([-\d.]+)")
    terrain_pat = re.compile(r"Mean episode terrain_level:\s*([-\d.]+)")

    iterations, rewards, terrain_levels = [], [], []
    current_iter = None
    current_reward = None

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_iter = iter_pat.search(line)
            if m_iter:
                current_iter = int(m_iter.group(1))
                continue

            if current_iter is not None:
                if current_reward is None:
                    m_rew = reward_pat.search(line)
                    if m_rew:
                        current_reward = float(m_rew.group(1))
                        continue
                
                if current_reward is not None:
                    m_terrain = terrain_pat.search(line)
                    if m_terrain:
                        iterations.append(current_iter)
                        rewards.append(current_reward)
                        terrain_levels.append(float(m_terrain.group(1)))
                        current_iter = None
                        current_reward = None
    return iterations, rewards, terrain_levels


def sample_every(iterations, rewards, terrain_levels, step=500):
    sel_iters, sel_rews, sel_terrains = [], [], []
    for it, rw, tl in zip(iterations, rewards, terrain_levels):
        if it % step == 0:
            sel_iters.append(it)
            sel_rews.append(rw)
            sel_terrains.append(tl)
    return sel_iters, sel_rews, sel_terrains


def main(log_path: str, step: int = 1):
    """绘制 Mean Reward‑vs‑Iteration 曲线.

    Parameters
    ----------
    log_path : str
        日志文件的路径（txt）
    step : int, optional
        采样间隔，默认 500
    """
    iterations, rewards, terrain_levels = parse_log(Path(log_path))
    if not iterations:
        raise RuntimeError("未在日志中找到 iteration / reward 数据，请检查格式。")

    iters_sampled, rews_sampled, terrains_sampled = sample_every(iterations, rewards, terrain_levels, step)

    # 找到最大值及其对应的迭代次数
    max_reward = max(rewards)
    max_reward_iter = iterations[rewards.index(max_reward)]
    max_reward_terrain = terrain_levels[rewards.index(max_reward)]

    max_terrain = max(terrain_levels)
    max_terrain_iter = iterations[terrain_levels.index(max_terrain)]
    max_terrain_reward = rewards[terrain_levels.index(max_terrain)]

    print(f"\n=== 统计信息 ===")
    print(f"最大奖励: {max_reward:.4f}, 迭代次数: {max_reward_iter}, 地形: {max_reward_terrain:.4f}")
    print(f"最大地形: {max_terrain:.4f}, 迭代次数: {max_terrain_iter}, 奖励: {max_terrain_reward:.4f}")
    print("="*40)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(iters_sampled, rews_sampled, marker="o", linewidth=1, color='blue', label='Mean Reward')
    # 标注最大奖励点
    ax1.scatter([max_reward_iter], [max_reward], color='red', s=100, marker='*', zorder=5, label=f'Max Reward: {max_reward:.4f}')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean Reward", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.plot(iters_sampled, terrains_sampled, marker="s", linewidth=1, color='green', label='Terrain Level')
    # 标注最大地形等级点
    ax2.scatter([max_terrain_iter], [max_terrain], color='orange', s=100, marker='*', zorder=5, label=f'Max Terrain: {max_terrain:.4f}')
    ax2.set_ylabel("Terrain Level", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    fig.suptitle(f"Training Progress (every {step} iters)")
    
    # 合并两个坐标轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
    
    plt.tight_layout()
    
    # 先保存图片到日志文件同一目录
    log_file = Path(log_path)
    output_path = log_file.parent / f"training_progress_{log_file.stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")
    
    # 尝试显示图表（如果有图形界面）
    try:
        plt.show()
    except Exception as e:
        print(f"无法显示图表窗口: {e}")
        print("图表已保存，请查看保存的文件。")
    finally:
        plt.close()


def find_latest_log(experiment_name: str) -> Path:
    """查找指定实验名称的最新日志文件。"""
    logs_dir = Path(LEGGED_GYM_LOGS_DIR) / experiment_name
    latest_log = None
    latest_time = 0
    
    if logs_dir.exists():
        for run_dir in logs_dir.iterdir():
            if run_dir.is_dir():
                for log_file in run_dir.glob('training_log_*.txt'):
                    file_time = log_file.stat().st_mtime
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_log = log_file
    
    if latest_log:
        return latest_log
    else:
        raise FileNotFoundError("未找到training_log文件")


if __name__ == "__main__":
    args = get_args()
    _, train_cfg = task_registry.get_cfgs(name=args.task)
    log_path = find_latest_log(train_cfg.runner.experiment_name)
    print(f"使用日志文件: {log_path}")
    main(str(log_path))
