#!/usr/bin/env python3
# Copyright (c) 2026-2027 zh
"""
内容：
    读取一个或多个 Go1 恢复训练目录内的 TensorBoard event 文件，对比绘制平均奖励和地形课程等级曲线。
    使用 TensorBoard fast data server 加载日志，每个标量标签最多保留 10,000 个点，仅导出 PDF。

用法：
    python tools/plot_reward_terrain_curves.py \
        --sample_step 10 \
        --input_dir logs/go1_recover_multi_critic/<run-a>/ logs/go1_recover_multi_critic/<run-b>/
"""

from __future__ import annotations

import argparse
import math
import time
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from tensorboard.data import provider, server_ingester
from tensorboard.util import grpc_util

plt.switch_backend("Agg")


# ---------- 配置 ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REWARD_TAG = "Train/mean_reward"
TERRAIN_LEVEL_TAG = "Episode/terrain_level"
FONT_FAMILY = "Times New Roman"
CURVE_COLORS = ("#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E", "#17BECF", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22")
FAST_SCALAR_LIMIT = 10_000
FAST_LOAD_TIMEOUT_SECONDS = 120.0
FAST_LOAD_STABLE_SECONDS = 5.0


@dataclass(frozen=True)
class ScalarSeries:
    """保存一组 TensorBoard 标量数据。"""

    steps: list[int]
    values: list[float]


# ---------- 数据读取 ----------

def resolve_run_directory(path: Path) -> Path:
    """将训练运行目录的绝对路径或仓库相对路径解析为存在的目录。"""
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Training run directory does not exist: {path}")
    return path


def load_training_series(run_directory: Path) -> tuple[ScalarSeries, ScalarSeries]:
    """通过 TensorBoard fast data server 加载奖励和地形等级序列。"""
    event_files = list(run_directory.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in: {run_directory}")
    ingester = server_ingester.SubprocessServerDataIngester(server_ingester.get_server_binary(), str(run_directory), reload_interval=0, channel_creds_type=grpc_util.ChannelCredsType.LOCAL, samples_per_plugin={"scalars": FAST_SCALAR_LIMIT})
    print(f"  Loading TensorBoard data (event files: {len(event_files)})...")
    ingester.start()
    data_provider = ingester.data_provider
    required_tags = {REWARD_TAG, TERRAIN_LEVEL_TAG}
    deadline = time.monotonic() + FAST_LOAD_TIMEOUT_SECONDS
    stable_since: float | None = None
    previous_max_steps: dict[str, int] | None = None
    while True:
        scalar_metadata = data_provider.list_scalars(None, experiment_id="", plugin_name="scalars")
        available_tags = {tag for tags in scalar_metadata.values() for tag in tags}
        if required_tags <= available_tags:
            max_steps = {tag: max(tags[tag].max_step for tags in scalar_metadata.values() if tag in tags) for tag in required_tags}
            if max_steps != previous_max_steps:
                previous_max_steps = max_steps
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= FAST_LOAD_STABLE_SECONDS:
                break
        if time.monotonic() >= deadline:
            missing_tags = sorted(required_tags - available_tags)
            detail = f"scalar tags {missing_tags}" if missing_tags else f"stable scalar data after steps {previous_max_steps}"
            raise TimeoutError(f"Timed out waiting for {detail} in: {run_directory}")
        time.sleep(0.1)

    scalar_data = data_provider.read_scalars(None, experiment_id="", plugin_name="scalars", downsample=FAST_SCALAR_LIMIT, run_tag_filter=provider.RunTagFilter(tags=required_tags))

    def build_series(tag: str) -> ScalarSeries:
        points = [point for tags in scalar_data.values() for point in tags.get(tag, [])]
        if not points:
            raise RuntimeError(f"Scalar tag '{tag}' has no data in: {run_directory}")
        points.sort(key=lambda point: point.step)
        return ScalarSeries([point.step for point in points], [point.value for point in points])

    reward_series = build_series(REWARD_TAG)
    terrain_series = build_series(TERRAIN_LEVEL_TAG)
    first_step = min(reward_series.steps[0], terrain_series.steps[0])
    last_step = max(reward_series.steps[-1], terrain_series.steps[-1])
    print(f"  Loaded points: reward={len(reward_series.steps):,}, terrain={len(terrain_series.steps):,}; steps={first_step:,}-{last_step:,}")
    return reward_series, terrain_series


def downsample_series(series: ScalarSeries, sample_step: int) -> ScalarSeries:
    """按固定间隔抽取真实数据点，并保留序列最后一点。"""
    indices = list(range(0, len(series.steps), sample_step))
    if indices[-1] != len(series.steps) - 1:
        indices.append(len(series.steps) - 1)
    return ScalarSeries([series.steps[index] for index in indices], [series.values[index] for index in indices])


# ---------- 绘图 ----------

def register_font(font_path: Path | None) -> str:
    """注册并返回绘图字体名称；默认严格要求 Times New Roman。"""
    if font_path is not None:
        if not font_path.is_file():
            raise FileNotFoundError(f"Font file does not exist: {font_path}")
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        try:
            font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY), fallback_to_default=False)
        except ValueError as error:
            raise RuntimeError("Times New Roman is not installed. Provide its regular font file with --font_path.") from error
        family = FONT_FAMILY
    return family


def configure_paper_style(font_family: str) -> None:
    """配置适合论文排版的 Matplotlib 样式。"""
    plt.rcParams.update({"font.family": font_family, "font.size": 9.0, "axes.labelsize": 10.0, "axes.linewidth": 0.8, "xtick.labelsize": 9.0, "ytick.labelsize": 9.0, "lines.solid_capstyle": "round", "path.simplify": False, "pdf.fonttype": 42, "pdf.use14corefonts": False})


def calculate_limits(values: list[float], margin: float, major_step: float, nonnegative: bool = False) -> tuple[float, float]:
    """按主刻度生成包含余量的纵轴范围。"""
    lower = math.floor((min(values) - margin) / major_step) * major_step
    upper = math.ceil((max(values) + margin) / major_step) * major_step
    if nonnegative:
        lower = max(0.0, lower)
    if lower == upper:
        upper = lower + major_step
    return lower, upper


def plot_series(labeled_series: list[tuple[str, ScalarSeries]], output_path: Path, y_label: str, y_limits: tuple[float, float], y_major_step: float) -> None:
    """绘制一组带图例的训练曲线并导出 PDF。"""
    fig, axis = plt.subplots(figsize=(7.16, 3.65))
    for index, (label, series) in enumerate(labeled_series):
        x_values = [step / 1000.0 for step in series.steps]
        axis.plot(x_values, series.values, color=CURVE_COLORS[index % len(CURVE_COLORS)], linewidth=0.9, alpha=0.88, label=label, zorder=3)

    x_upper = max(1.0, max(math.ceil(series.steps[-1] / 1000.0) for _, series in labeled_series))
    axis.set_xlim(0.0, x_upper)
    axis.set_ylim(*y_limits)
    axis.set_xlabel("Training iteration (×10³)")
    axis.set_ylabel(y_label)
    axis.xaxis.set_major_locator(MultipleLocator(10.0 if x_upper >= 50.0 else 2.0))
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_major_locator(MultipleLocator(y_major_step))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.grid(axis="y", which="major", color="#D0D0D0", linewidth=0.55, linestyle="--", alpha=0.75)
    axis.tick_params(axis="both", which="major", direction="in", top=True, right=True, width=0.8, length=4.0)
    axis.tick_params(axis="both", which="minor", direction="in", top=True, right=True, width=0.6, length=2.2)
    axis.legend(frameon=False, loc="lower right")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------- 主程序 ----------

def parse_args() -> argparse.Namespace:
    """解析训练运行目录、输出和绘图参数。"""
    parser = argparse.ArgumentParser(description="Compare reward and terrain-level curves from one or more training run directories")
    parser.add_argument("--input_dir", type=Path, nargs="+", required=True, help="One or more absolute or project-relative training run directories")
    parser.add_argument("--output_dir", type=Path, help="PDF output directory; defaults to the last input directory/output/plot_reward_terrain_curves")
    parser.add_argument("--font_path", type=Path, help="Times New Roman regular font file, for example times.ttf")
    parser.add_argument("--sample_step", type=int, default=10, help="Downsampling interval; 1 keeps all points, default: 10")
    args = parser.parse_args()
    if args.sample_step < 1:
        parser.error("--sample_step must be a positive integer")
    return args


def main() -> None:
    """读取一个或多个运行目录的训练数据并生成对比曲线 PDF。"""
    args = parse_args()
    run_directories = [resolve_run_directory(path) for path in args.input_dir]
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_directories[-1] / "output" / Path(__file__).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_series: list[tuple[str, ScalarSeries]] = []
    terrain_series: list[tuple[str, ScalarSeries]] = []
    run_count = len(run_directories)
    print(f"Training runs: {run_count}")
    for index, run_directory in enumerate(run_directories, start=1):
        print(f"[{index}/{run_count}] {run_directory.name}")
        reward, terrain = load_training_series(run_directory)
        reward_series.append((run_directory.name, downsample_series(reward, args.sample_step)))
        terrain_series.append((run_directory.name, downsample_series(terrain, args.sample_step)))

    font_family = register_font(args.font_path)
    configure_paper_style(font_family)
    reward_output = output_dir / "reward.pdf"
    terrain_output = output_dir / "terrain_level.pdf"
    reward_values = [value for _, series in reward_series for value in series.values]
    terrain_values = [value for _, series in terrain_series for value in series.values]
    plot_series(reward_series, reward_output, "Mean episode reward", calculate_limits(reward_values, margin=1.0, major_step=10.0), 10.0)
    plot_series(terrain_series, terrain_output, "Mean terrain level", calculate_limits(terrain_values, margin=0.2, major_step=1.0, nonnegative=True), 1.0)
    print(f"Font: {font_family}")
    print(f"Downsampling interval: {args.sample_step}")
    print(f"Output directory: {output_dir}")
    print(f"  - {reward_output.name}")
    print(f"  - {terrain_output.name}")


if __name__ == "__main__":
    main()
