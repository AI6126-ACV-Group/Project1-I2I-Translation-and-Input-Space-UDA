import argparse
import os
import shutil
from typing import Dict, List

from office31_uda.common import (
    ensure_dir,
    load_manifest_records,
    make_cycle_stem,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Amazon/Webcam manifests as CycleGAN trainA/trainB/testA/testB folders."
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--source-manifests",
        type=str,
        nargs="+",
        required=True,
        help="Manifests to stage as domain A. Pass amazon_train.json amazon_val.json.",
    )
    parser.add_argument(
        "--target-manifests",
        type=str,
        nargs="+",
        required=True,
        help="Manifests to stage as domain B. Pass webcam_train.json.",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        default="copy",
        choices=["copy", "hardlink", "symlink"],
    )
    parser.add_argument("--limit-test-b", type=int, default=128)
    return parser.parse_args()


def stage_file(src: str, dst: str, mode: str) -> None:
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    shutil.copy2(src, dst)


def stage_records(
    records: List[Dict],
    data_root: str,
    split_name: str,
    out_dir: str,
    mode: str,
    start_index: int,
) -> List[Dict]:
    staged: List[Dict] = []
    for offset, record in enumerate(records):
        index = start_index + offset
        stem = make_cycle_stem(index=index, record=record)
        src_path = os.path.join(data_root, record["rel_path"])
        dst_rel = f"{stem}{os.path.splitext(record['filename'])[1].lower()}"
        dst_path = os.path.join(out_dir, dst_rel)
        stage_file(src_path, dst_path, mode=mode)
        staged.append(
            {
                **record,
                "split_name": split_name,
                "staged_name": os.path.basename(dst_path),
                "staged_rel_path": os.path.relpath(dst_path, os.path.dirname(out_dir)).replace("\\", "/"),
                "staged_stem": stem,
            }
        )
    return staged


def main() -> None:
    args = parse_args()

    train_a_dir = os.path.join(args.output_dir, "trainA")
    train_b_dir = os.path.join(args.output_dir, "trainB")
    test_a_dir = os.path.join(args.output_dir, "testA")
    test_b_dir = os.path.join(args.output_dir, "testB")
    meta_dir = os.path.join(args.output_dir, "metadata")
    for path in [train_a_dir, train_b_dir, test_a_dir, test_b_dir, meta_dir]:
        ensure_dir(path)

    staged_train_a: List[Dict] = []
    staged_test_a: List[Dict] = []
    next_index = 0
    for manifest_path in args.source_manifests:
        records = load_manifest_records(manifest_path)
        split_name = os.path.splitext(os.path.basename(manifest_path))[0]
        staged_train_a.extend(
            stage_records(
                records=records,
                data_root=args.data_root,
                split_name=split_name,
                out_dir=train_a_dir,
                mode=args.copy_mode,
                start_index=next_index,
            )
        )
        staged_test_a.extend(
            stage_records(
                records=records,
                data_root=args.data_root,
                split_name=split_name,
                out_dir=test_a_dir,
                mode=args.copy_mode,
                start_index=next_index,
            )
        )
        next_index += len(records)

    staged_train_b: List[Dict] = []
    next_index = 0
    target_all: List[Dict] = []
    for manifest_path in args.target_manifests:
        records = load_manifest_records(manifest_path)
        split_name = os.path.splitext(os.path.basename(manifest_path))[0]
        target_all.extend(records)
        staged_train_b.extend(
            stage_records(
                records=records,
                data_root=args.data_root,
                split_name=split_name,
                out_dir=train_b_dir,
                mode=args.copy_mode,
                start_index=next_index,
            )
        )
        next_index += len(records)

    staged_test_b = stage_records(
        records=target_all[: args.limit_test_b] if args.limit_test_b > 0 else target_all,
        data_root=args.data_root,
        split_name="preview_target",
        out_dir=test_b_dir,
        mode=args.copy_mode,
        start_index=0,
    )

    save_json(
        os.path.join(meta_dir, "testA_index.json"),
        {
            "records": staged_test_a,
            "data_root": os.path.abspath(args.data_root),
        },
    )
    save_json(
        os.path.join(meta_dir, "trainA_index.json"),
        {
            "records": staged_train_a,
            "data_root": os.path.abspath(args.data_root),
        },
    )
    save_json(
        os.path.join(meta_dir, "trainB_index.json"),
        {
            "records": staged_train_b,
            "data_root": os.path.abspath(args.data_root),
        },
    )
    save_json(
        os.path.join(meta_dir, "testB_index.json"),
        {
            "records": staged_test_b,
            "data_root": os.path.abspath(args.data_root),
        },
    )

    print("Prepared CycleGAN dataset at:", os.path.abspath(args.output_dir))
    print("trainA:", len(staged_train_a))
    print("trainB:", len(staged_train_b))
    print("testA:", len(staged_test_a))
    print("testB:", len(staged_test_b))


if __name__ == "__main__":
    main()
