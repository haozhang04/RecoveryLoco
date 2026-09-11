# Copyright (c) 2026-2027 zh
"""
内容：
    按 GPU 数量均匀分配训练任务，每张 GPU 最多同时运行两个任务。
    超出并发容量的任务自动排队；异常任务在首轮结束后以每张 GPU 一个任务的方式重试一次。
    异常日志单独重命名保存，重试使用标准日志文件名，避免影响训练曲线解析。
    模型和训练进程日志统一归档到输出根目录的 train_data/ 和 training_logs/<timestamp>/。

用法：
    python tools/train_tasks.py --gpu_ids 0 1
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path


# ---------- 配置 ----------

OUTPUT_TIME_FORMAT = "%Y%m%d_%H%M%S_%f"
OUTPUT_ROOT_PREFIX = "wo-"
TRAIN_DATA_DIRNAME = "train_data"
TRAINING_LOGS_DIRNAME = "training_logs"
TASKS = [
    "go1_recover_multi_critic",
    "go1_recover_multi_critic_him",
    "go1_recover_multi_critic_vae",
    "go1_recover_single_critic",
    "go1_recover_multi_critic_without_symmetry",
    "go1_recover_multi_critic_without_curriculum",
    "go1_recover_multi_critic_without_estimator",
    "go1_recover_multi_critic_without_upright_gate",
]
TRAIN_ARGS = ["--max_iterations", "20000", "--seed", "1"]
MAX_TASKS_PER_GPU = 2
RETRY_TASKS_PER_GPU = 1
TERMINATION_TIMEOUT_S = 10


# ---------- 输出目录 ----------

def create_output_root(logs_root):
    """创建并返回唯一的 logs/wo-<timestamp> 训练输出根目录。"""
    output_root = logs_root / f"{OUTPUT_ROOT_PREFIX}{datetime.now().strftime(OUTPUT_TIME_FORMAT)}"
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def create_training_log_dir(output_root):
    """在统一输出根目录下创建并返回本次训练的进程日志目录。"""
    log_dir = output_root / TRAINING_LOGS_DIRNAME / datetime.now().strftime(OUTPUT_TIME_FORMAT)
    log_dir.mkdir(parents=True, exist_ok=False)
    return log_dir


# ---------- helpers ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Distribute training tasks evenly across GPUs")
    parser.add_argument("--gpu_ids", type=int, nargs="+", required=True, help="GPU IDs used for training, for example: --gpu_ids 0 1")
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Unified output root containing train_data and training_logs; defaults to a new <project_root>/logs/wo-<timestamp>",
    )
    return parser.parse_args()


def assign_tasks(tasks, gpu_ids):
    if not tasks:
        raise ValueError("TASKS cannot be empty")
    if not gpu_ids:
        raise ValueError("GPU_IDS cannot be empty")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU_IDS cannot contain duplicates")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("GPU_IDS cannot contain negative values")
    if MAX_TASKS_PER_GPU <= 0:
        raise ValueError("MAX_TASKS_PER_GPU must be greater than zero")

    assignments = {gpu_id: [] for gpu_id in gpu_ids}
    for index, task in enumerate(tasks):
        assignments[gpu_ids[index % len(gpu_ids)]].append((index, task))
    return assignments


def build_environment(project_root, gpu_id, train_data_dir=None):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    environment["PYTHONPATH"] = f"{project_root / 'legged_gym'}{os.pathsep}{environment.get('PYTHONPATH', '')}"
    if train_data_dir is not None:
        environment["LEGGED_GYM_TRAIN_DATA_ROOT"] = str(train_data_dir)
    conda_prefix = environment.get("CONDA_PREFIX")
    if conda_prefix:
        environment["LD_LIBRARY_PATH"] = f"{Path(conda_prefix) / 'lib'}{os.pathsep}{environment.get('LD_LIBRARY_PATH', '')}"
    return environment


def start_task(project_root, train_script, train_data_dir, log_dir, index, task, gpu_id, attempt):
    log_path = log_dir / f"{index + 1:02d}_{task}.log"
    log_file = log_path.open("w", encoding="utf-8")
    command = [sys.executable, str(train_script), "--headless", "--task", task, "--rl_device", "cuda:0", *TRAIN_ARGS]
    try:
        process = subprocess.Popen(command, cwd=project_root, env=build_environment(project_root, gpu_id, train_data_dir), stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        log_file.close()
        raise
    print(f"Started task '{task}' attempt {attempt} on GPU {gpu_id}, log: {log_path}")
    return process, index, task, gpu_id, log_file


def archive_failed_log(log_dir, index, task, attempt):
    log_path = log_dir / f"{index + 1:02d}_{task}.log"
    failed_log_path = log_dir / f"{index + 1:02d}_{task}_attempt{attempt}_failed.log"
    log_path.rename(failed_log_path)
    print(f"Archived failed task log: {failed_log_path}")
    return failed_log_path


def stop_processes(processes):
    for process, _, _, _, _ in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + TERMINATION_TIMEOUT_S
    for process, _, _, _, log_file in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        log_file.close()


def handle_termination(_, __):
    raise KeyboardInterrupt


def run_task_queues(project_root, train_script, train_data_dir, log_dir, gpu_ids, task_queues, tasks_per_gpu, attempt):
    active_processes = []
    failed_tasks = []
    try:
        for _ in range(tasks_per_gpu):
            for gpu_id in gpu_ids:
                if task_queues[gpu_id]:
                    index, task = task_queues[gpu_id].popleft()
                    active_processes.append(start_task(project_root, train_script, train_data_dir, log_dir, index, task, gpu_id, attempt))

        while active_processes:
            task_finished = False
            for process_info in active_processes.copy():
                process, index, task, gpu_id, log_file = process_info
                return_code = process.poll()
                if return_code is None:
                    continue

                task_finished = True
                active_processes.remove(process_info)
                log_file.close()
                if return_code != 0:
                    archive_failed_log(log_dir, index, task, attempt)
                    failed_tasks.append((index, task, gpu_id, return_code))
                print(f"Finished task '{task}' attempt {attempt} on GPU {gpu_id} with exit code {return_code}")

                if task_queues[gpu_id]:
                    next_index, next_task = task_queues[gpu_id].popleft()
                    active_processes.append(start_task(project_root, train_script, train_data_dir, log_dir, next_index, next_task, gpu_id, attempt))
            if active_processes and not task_finished:
                time.sleep(1)
    except BaseException:
        stop_processes(active_processes)
        raise
    return failed_tasks


def build_retry_queues(failed_tasks, gpu_ids):
    retry_queues = {gpu_id: deque() for gpu_id in gpu_ids}
    for retry_index, (index, task, _, _) in enumerate(failed_tasks):
        gpu_id = gpu_ids[retry_index % len(gpu_ids)]
        retry_queues[gpu_id].append((index, task))
    return retry_queues


# ---------- main ----------

def main():
    signal.signal(signal.SIGTERM, handle_termination)
    args = parse_args()
    gpu_ids = args.gpu_ids
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else create_output_root(project_root / "logs").resolve()
    train_data_dir = output_dir / TRAIN_DATA_DIRNAME
    train_script = project_root / "legged_gym" / "legged_gym" / "scripts" / "train.py"
    assignments = assign_tasks(TASKS, gpu_ids)
    train_data_dir.mkdir(parents=True, exist_ok=True)
    log_dir = create_training_log_dir(output_dir)
    task_queues = {gpu_id: deque(tasks) for gpu_id, tasks in assignments.items()}

    print(f"Unified output root: {output_dir}")
    print(f"Training data root: {train_data_dir}")
    print(f"Training process logs: {log_dir}")
    for gpu_id, tasks in assignments.items():
        print(f"GPU {gpu_id}: {[task for _, task in tasks]}")

    retried_task_count = 0
    try:
        failed_tasks = run_task_queues(project_root, train_script, train_data_dir, log_dir, gpu_ids, task_queues, MAX_TASKS_PER_GPU, attempt=1)
        if failed_tasks:
            retried_task_count = len(failed_tasks)
            print(f"Retrying {retried_task_count} failed task(s) after all initial tasks have finished")
            retry_queues = build_retry_queues(failed_tasks, gpu_ids)
            failed_tasks = run_task_queues(project_root, train_script, train_data_dir, log_dir, gpu_ids, retry_queues, RETRY_TASKS_PER_GPU, attempt=2)
    except KeyboardInterrupt:
        print("Interrupted, stopping all training tasks...")
        return 130
    except BaseException:
        print("Unexpected error, stopping all training tasks...")
        raise

    if failed_tasks:
        for _, task, gpu_id, return_code in failed_tasks:
            print(f"Task '{task}' failed again on GPU {gpu_id} with exit code {return_code}; no more retries")
        return 1
    if retried_task_count:
        print(f"All training tasks completed successfully after retrying {retried_task_count} task(s)")
    else:
        print("All training tasks completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
