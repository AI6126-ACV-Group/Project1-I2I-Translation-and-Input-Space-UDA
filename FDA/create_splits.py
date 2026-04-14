import argparse
import os

from FDA.common import ensure_dir, set_seed
from FDA.data import (
    TASKS,
    build_digit_splits,
    build_imagefolder_splits,
    get_task_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible FDA split manifests.")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASKS))
    parser.add_argument("--data-root", type=str, default="./Datasets")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--source-val-ratio", type=float, default=-1.0)
    parser.add_argument("--target-test-ratio", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = get_task_config(args.task)
    set_seed(args.seed)

    output_dir = args.output_dir or os.path.join("./FDA/outputs/splits", task.name)
    ensure_dir(output_dir)

    source_val_ratio = None if args.source_val_ratio <= 0 else args.source_val_ratio
    target_test_ratio = None if args.target_test_ratio <= 0 else args.target_test_ratio

    if task.task_type == "digits":
        build_digit_splits(
            task=task,
            data_root=args.data_root,
            output_dir=output_dir,
            source_val_ratio=source_val_ratio,
            seed=args.seed,
        )
    else:
        build_imagefolder_splits(
            task=task,
            data_root=args.data_root,
            output_dir=output_dir,
            source_val_ratio=source_val_ratio,
            target_test_ratio=target_test_ratio,
            seed=args.seed,
        )

    print(f"Saved FDA split manifests to: {os.path.abspath(output_dir)}")
    print(f"Task: {task.name}")
    print(f"Data root: {os.path.abspath(args.data_root)}")


if __name__ == "__main__":
    main()
