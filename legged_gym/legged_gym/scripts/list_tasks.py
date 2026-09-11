#!/usr/bin/env python3

import argparse

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils.task_registry import task_registry


def _group_task_name(name):
    prefix = name.split("_", 1)[0]
    return prefix.upper() if prefix.islower() else prefix


def _print_table(groups):
    title = "Registered Legged Gym Tasks"
    total = sum(len(names) for names in groups.values())
    width = max(len(title), 44)

    print()
    print("=" * width)
    print(title.center(width))
    print("=" * width)
    print(f"Total: {total}")

    if total == 0:
        print()
        print("No registered tasks matched.")
        print()
        return

    for group in sorted(groups.keys()):
        names = sorted(groups[group])
        print()
        print(f"[{group}]")
        for index, name in enumerate(names, start=1):
            print(f"  {index:>2}. {name}")
    print()


def main():
    parser = argparse.ArgumentParser(description="List registered legged_gym tasks.")
    parser.add_argument(
        "--f",
        dest="filter",
        default="",
        help="Only show tasks containing this text.",
    )
    args = parser.parse_args()

    task_names = sorted(task_registry.task_classes.keys())
    if args.filter:
        task_names = [name for name in task_names if args.filter.lower() in name.lower()]

    groups = {}
    for name in task_names:
        groups.setdefault(_group_task_name(name), []).append(name)

    _print_table(groups)


if __name__ == "__main__":
    main()
