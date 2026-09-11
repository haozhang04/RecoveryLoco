#!/usr/bin/env python3
# Copyright (c) 2026-2027 zh
"""
内容：
    解析恢复训练的消融实验日志，绘制论文用平均奖励和地形课程等级曲线。
    全量、基线和消融对比图均对日志原始数据进行等间隔降采样，不进行平滑或插值，仅导出 PDF。

用法：
    python tools/train_tasks_plot_reward_terrain_curves.py \
        --sample_step 10 \
        --input_dir logs/wo-<timestamp>/
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


# ---------- 配置 ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_LOGS_DIRNAME = "training_logs"
FONT_FAMILY = "Times New Roman"
TRAINING_BATCH_RE = re.compile(r"\d{8}_\d{6}_\d{6}")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)/")
MEAN_REWARD_RE = re.compile(r"Mean reward:\s*([-+]?\d+(?:\.\d+)?)")
TERRAIN_LEVEL_RE = re.compile(r"Mean episode terrain_level:\s*([-+]?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class ExperimentStyle:
    """定义单个消融实验在论文图中的名称和视觉编码。"""

    filename: str
    label: str
    color: str
    linestyle: object
    linewidth: float = 0.8
    zorder: int = 3


@dataclass(frozen=True)
class TrainingCurve:
    """保存一条训练曲线及其绘图样式。"""

    style: ExperimentStyle
    iterations: list[int]
    mean_rewards: list[float]
    terrain_levels: list[float]


# 延续原代码的配色，其中核心方法和已有消融项严格使用原颜色。
OURS_STYLE = ExperimentStyle("01_go1_recover_multi_critic.log", "Ours", "#D62728", "-", linewidth=1.15, zorder=10)
BASELINE_STYLES: tuple[ExperimentStyle, ...] = (
    ExperimentStyle("02_go1_recover_multi_critic_him.log", "HIMLoco", "#1F77B4", "--"),
    ExperimentStyle("03_go1_recover_multi_critic_vae.log", "Dreamwaq", "#17BECF", "-."),
)
ABLATION_STYLES: tuple[ExperimentStyle, ...] = (
    ExperimentStyle("04_go1_recover_single_critic.log", "Single critic", "#7F7F7F", (0, (5, 1))),
    ExperimentStyle("05_go1_recover_multi_critic_without_symmetry.log", "w/o symmetry", "#E377C2", (0, (3, 1, 1, 1))),
    ExperimentStyle("06_go1_recover_multi_critic_without_curriculum.log", "w/o curriculum", "#8C6BB1", (0, (1, 1))),
    ExperimentStyle("07_go1_recover_multi_critic_without_estimator.log", "w/o estimator", "#D95F02", (0, (5, 2, 1, 2))),
    ExperimentStyle("08_go1_recover_multi_critic_without_upright_gate.log", "w/o upright gate", "#1B9E77", (0, (2, 1))),
)


# ---------- 日志解析 ----------

def resolve_project_path(path: Path) -> Path:
    """将仓库相对路径或绝对路径解析为规范路径。"""
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def find_latest_log_batch(training_logs_dir: Path) -> Path:
    """返回 training_logs 下名称最新且包含日志文件的训练批次。"""
    candidates = [
        path
        for path in training_logs_dir.iterdir()
        if path.is_dir() and TRAINING_BATCH_RE.fullmatch(path.name) and any(path.glob("*.log"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No training log batch found in: {training_logs_dir}")
    return max(candidates, key=lambda path: path.name)


def resolve_training_log_dir(input_dir: Path) -> Path:
    """将统一输出根目录、training_logs 或训练日志批次解析为具体批次目录。"""
    input_dir = resolve_project_path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if any(input_dir.glob("*.log")):
        return input_dir

    training_logs_dir = input_dir if input_dir.name == TRAINING_LOGS_DIRNAME else input_dir / TRAINING_LOGS_DIRNAME
    if not training_logs_dir.is_dir():
        raise FileNotFoundError(f"Training logs directory does not exist: {training_logs_dir}")
    return find_latest_log_batch(training_logs_dir)


def parse_training_log(path: Path) -> tuple[list[int], list[float], list[float]]:
    """从单个日志中提取迭代次数、平均奖励和平均地形等级。"""
    iterations: list[int] = []
    mean_rewards: list[float] = []
    terrain_levels: list[float] = []
    current_iteration: int | None = None
    current_reward: float | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for raw_line in stream:
            line = ANSI_ESCAPE_RE.sub("", raw_line)
            iteration_match = ITERATION_RE.search(line)
            if iteration_match:
                current_iteration = int(iteration_match.group(1))
                current_reward = None
                continue

            reward_match = MEAN_REWARD_RE.search(line)
            if reward_match is not None and current_iteration is not None:
                current_reward = float(reward_match.group(1))
                continue

            terrain_match = TERRAIN_LEVEL_RE.search(line)
            if terrain_match is None or current_iteration is None or current_reward is None:
                continue
            iterations.append(current_iteration)
            mean_rewards.append(current_reward)
            terrain_levels.append(float(terrain_match.group(1)))
            current_iteration = None
            current_reward = None

    return iterations, mean_rewards, terrain_levels


def load_training_curves(logs_dir: Path) -> list[TrainingCurve]:
    """按固定论文顺序读取存在完整指标的实验曲线。"""
    curves: list[TrainingCurve] = []
    for style in (OURS_STYLE, *BASELINE_STYLES, *ABLATION_STYLES):
        log_path = logs_dir / style.filename
        if not log_path.exists():
            print(f"Warning: log file not found, skipped: {log_path}")
            continue

        iterations, mean_rewards, terrain_levels = parse_training_log(log_path)
        if not iterations:
            print(f"Warning: no complete reward and terrain-level data found, skipped: {log_path}")
            continue

        curves.append(TrainingCurve(style, iterations, mean_rewards, terrain_levels))
        print(
            f"Loaded {style.label}: {len(iterations)} points, iteration {iterations[0]}-{iterations[-1]}, "
            f"final reward {mean_rewards[-1]:.4f}, final level {terrain_levels[-1]:.4f}"
        )

    if not curves:
        raise RuntimeError(f"No valid training curves found in: {logs_dir}")
    return curves


def select_training_curves(curves: list[TrainingCurve], styles: tuple[ExperimentStyle, ...]) -> list[TrainingCurve]:
    """按给定样式顺序筛选训练曲线。"""
    curves_by_filename = {curve.style.filename: curve for curve in curves}
    selected = [curves_by_filename[style.filename] for style in styles if style.filename in curves_by_filename]
    if not selected:
        raise RuntimeError("No valid training curves found for comparison group.")
    return selected


# ---------- 数据抽样 ----------

def downsample_series(iterations: list[int], values: list[float], step: int) -> tuple[list[int], list[float]]:
    """按固定间隔抽取真实数据点，并保留序列最后一个点。"""
    if len(iterations) != len(values):
        raise ValueError("Iteration and value sequences must have the same length.")
    if step < 1:
        raise ValueError("Downsampling step must be a positive integer.")
    if not iterations:
        return [], []

    indices = list(range(0, len(iterations), step))
    if indices[-1] != len(iterations) - 1:
        indices.append(len(iterations) - 1)
    return [iterations[index] for index in indices], [values[index] for index in indices]


# ---------- 绘图 ----------

def register_font(font_path: Path | None) -> str:
    """注册并返回绘图字体名称；默认严格要求 Times New Roman。"""
    if font_path is not None:
        if not font_path.is_file():
            raise FileNotFoundError(f"Font file not found: {font_path}")
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        try:
            font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY), fallback_to_default=False)
        except ValueError as error:
            raise RuntimeError("Times New Roman is not installed. Provide its regular font file with --font_path.") from error
        family = FONT_FAMILY

    print(f"Using font: {family}")
    return family


def configure_paper_style(font_family: str) -> None:
    """配置适合双栏论文宽度的 Matplotlib 全局样式。"""
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "path.simplify": False,
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
        }
    )


def plot_training_metric(
    curves: list[TrainingCurve],
    output_path: Path,
    metric_name: str,
    y_label: str,
    y_limits: tuple[float, float],
    y_major_step: float,
    metadata_title: str,
    sample_step: int,
) -> None:
    """对日志原始序列等间隔降采样后绘制单项训练指标。"""
    fig, axis = plt.subplots(figsize=(7.16, 3.65))
    max_iteration = max(curve.iterations[-1] for curve in curves)

    for curve in curves:
        sampled_iterations, sampled_values = downsample_series(curve.iterations, getattr(curve, metric_name), sample_step)
        x_values = [iteration / 1000.0 for iteration in sampled_iterations]
        axis.plot(
            x_values,
            sampled_values,
            color=curve.style.color,
            linestyle=curve.style.linestyle,
            linewidth=curve.style.linewidth,
            alpha=0.82,
            label=curve.style.label,
            zorder=curve.style.zorder,
        )

    x_upper = max(1.0, math.ceil(max_iteration / 1000.0))
    axis.set_xlim(0.0, x_upper)
    axis.set_ylim(*y_limits)
    axis.set_xlabel("Training iteration (×10³)")
    axis.set_ylabel(y_label)
    axis.xaxis.set_major_locator(MultipleLocator(2.0 if x_upper >= 8.0 else 1.0))
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_major_locator(MultipleLocator(y_major_step))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.grid(axis="y", which="major", color="#D0D0D0", linewidth=0.55, linestyle="--", alpha=0.75)
    axis.tick_params(axis="both", which="major", direction="in", top=True, right=True, width=0.8, length=4.0)
    axis.tick_params(axis="both", which="minor", direction="in", top=True, right=True, width=0.6, length=2.2)
    axis.legend(
        loc="lower right",
        ncol=1,
        frameon=False,
        handlelength=2.8,
        handletextpad=0.6,
        borderaxespad=0.7,
    )

    fig.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"Title": metadata_title, "Creator": "Matplotlib"},
    )
    plt.close(fig)


def plot_terrain_levels(curves: list[TrainingCurve], output_path: Path, sample_step: int) -> None:
    """绘制经过等间隔降采样且未经平滑的地形课程等级曲线。"""
    max_level = max(max(curve.terrain_levels) for curve in curves)
    y_upper = max(1.0, math.ceil((max_level + 0.2) * 2.0) / 2.0)
    plot_training_metric(
        curves,
        output_path,
        "terrain_levels",
        "Mean terrain level",
        (0.0, y_upper),
        1.0,
        "Terrain curriculum progression",
        sample_step,
    )


def plot_mean_rewards(curves: list[TrainingCurve], output_path: Path, sample_step: int) -> None:
    """绘制经过等间隔降采样且未经平滑的平均奖励曲线。"""
    sampled_reward_series = [downsample_series(curve.iterations, curve.mean_rewards, sample_step)[1] for curve in curves]
    min_reward = min(min(rewards) for rewards in sampled_reward_series)
    max_reward = max(max(rewards) for rewards in sampled_reward_series)
    y_lower = math.floor((min_reward - 1.0) / 5.0) * 5.0
    y_upper = math.ceil((max_reward + 1.0) / 5.0) * 5.0
    plot_training_metric(
        curves,
        output_path,
        "mean_rewards",
        "Mean episode reward",
        (y_lower, y_upper),
        10.0,
        "Mean episode reward progression",
        sample_step,
    )


def write_comparison_pdfs(
    curves: list[TrainingCurve], output_dir: Path, filename_prefix: str, sample_step: int
) -> tuple[Path, Path]:
    """输出指定曲线组的地形等级和奖励 PDF。"""
    terrain_output = output_dir / f"{filename_prefix}terrain_level.pdf"
    reward_output = output_dir / f"{filename_prefix}reward.pdf"
    plot_terrain_levels(curves, terrain_output, sample_step)
    plot_mean_rewards(curves, reward_output, sample_step)
    return terrain_output, reward_output


# ---------- 主程序 ----------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="等间隔降采样后绘制未经平滑的论文用平均奖励和地形课程等级曲线，仅导出 PDF。")
    parser.add_argument("--input_dir", type=Path, required=True, help="统一输出根目录、training_logs 目录或具体训练日志批次。")
    parser.add_argument("--output_dir", type=Path, help="六个 PDF 的输出目录；默认写入训练日志批次的 output/脚本名称。")
    parser.add_argument("--font_path", type=Path, help="Times New Roman 常规字体文件路径，例如 times.ttf。")
    parser.add_argument("--sample_step", type=int, default=10, help="等间隔降采样步长；1 表示保留全部数据，默认 10。")
    args = parser.parse_args()
    if args.sample_step < 1:
        parser.error("--sample_step must be a positive integer")
    return args


def main() -> None:
    """加载训练数据并生成全量、基线和消融对比 PDF。"""
    args = parse_args()
    logs_dir = resolve_training_log_dir(args.input_dir)
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else logs_dir / "output" / Path(__file__).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training log directory: {logs_dir}")
    curves = load_training_curves(logs_dir)
    baseline_curves = select_training_curves(curves, (OURS_STYLE, *BASELINE_STYLES))
    wo_curves = select_training_curves(curves, (OURS_STYLE, *ABLATION_STYLES))
    font_family = register_font(args.font_path)
    configure_paper_style(font_family)
    print(f"Downsampling step: {args.sample_step}")
    output_paths = [
        *write_comparison_pdfs(curves, output_dir, "", args.sample_step),
        *write_comparison_pdfs(baseline_curves, output_dir, "baseline_", args.sample_step),
        *write_comparison_pdfs(wo_curves, output_dir, "wo_", args.sample_step),
    ]
    for output_path in output_paths:
        print(f"Wrote PDF: {output_path.resolve()}")


if __name__ == "__main__":
    main()
