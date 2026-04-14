import argparse
import os
import re
from typing import Dict, List

from torchvision.transforms import functional as TF

from FDA.common import ensure_dir, save_json, set_seed, setup_logging
from FDA.data import TASKS, RecordAccessor, get_task_config, load_manifest_records
from FDA.fourier import apply_fda, prepare_image_for_fda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FDA-translated source images.")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASKS))
    parser.add_argument("--data-root", type=str, default="./Datasets")
    parser.add_argument("--split-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--fda-image-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def record_reference(record: Dict) -> str:
    if "rel_path" in record:
        return record["rel_path"]
    return f"{record['dataset']}:{record['split']}:{record['index']}"


def make_output_rel_path(split_name: str, record: Dict, item_idx: int) -> str:
    class_name = sanitize_name(record["class_name"])
    stem = sanitize_name(os.path.splitext(record.get("filename", str(item_idx)))[0])
    return f"images/{split_name}/{class_name}/{item_idx:06d}__{stem}.png"


def export_split(
    split_name: str,
    source_records: List[Dict],
    target_records: List[Dict],
    source_accessor: RecordAccessor,
    target_accessor: RecordAccessor,
    output_dir: str,
    image_mode: str,
    image_size: int,
    beta: float,
    seed: int,
    skip_existing: bool,
) -> List[Dict]:
    rng = __import__("random").Random(seed)
    exported_records: List[Dict] = []

    for item_idx, source_record in enumerate(source_records):
        target_record = rng.choice(target_records)
        rel_path = make_output_rel_path(split_name, source_record, item_idx)
        abs_path = os.path.join(output_dir, rel_path)
        ensure_dir(os.path.dirname(abs_path))

        if not (skip_existing and os.path.exists(abs_path)):
            source_image = source_accessor.load_image(source_record)
            target_image = target_accessor.load_image(target_record)
            source_tensor = prepare_image_for_fda(source_image, image_size=image_size, mode=image_mode)
            target_tensor = prepare_image_for_fda(target_image, image_size=image_size, mode=image_mode)
            mixed_tensor = apply_fda(source_tensor, target_tensor, beta=beta)
            TF.to_pil_image(mixed_tensor).save(abs_path)

        exported_records.append(
            {
                "rel_path": rel_path.replace("\\", "/"),
                "class_idx": int(source_record["class_idx"]),
                "class_name": source_record["class_name"],
                "image_mode": image_mode,
                "source_reference": record_reference(source_record),
                "target_reference": record_reference(target_record),
                "filename": os.path.basename(rel_path),
            }
        )

        if (item_idx + 1) % 500 == 0 or item_idx + 1 == len(source_records):
            print(f"[{split_name}] exported {item_idx + 1}/{len(source_records)}")

    return exported_records


def main() -> None:
    args = parse_args()
    task = get_task_config(args.task)
    set_seed(args.seed)

    split_dir = args.split_dir or os.path.join("./FDA/outputs/splits", task.name)
    output_dir = args.output_dir or os.path.join("./FDA/outputs/exports", task.name)
    ensure_dir(output_dir)
    setup_logging(output_dir, filename="export.log")

    fda_image_size = args.fda_image_size or task.export_image_size
    image_mode = "L" if task.task_type == "digits" else "RGB"

    print(f"Task: {task.name}")
    print(f"Data root: {os.path.abspath(args.data_root)}")
    print(f"Split dir: {os.path.abspath(split_dir)}")
    print(f"Output dir: {os.path.abspath(output_dir)}")
    print(f"beta: {args.beta}")
    print(f"fda_image_size: {fda_image_size}")
    print(f"image_mode: {image_mode}")

    source_train = load_manifest_records(os.path.join(split_dir, "source_train.json"))
    source_val = load_manifest_records(os.path.join(split_dir, "source_val.json"))
    target_train = load_manifest_records(os.path.join(split_dir, "target_train.json"))

    source_accessor = RecordAccessor(task=task, data_root=args.data_root)
    target_accessor = RecordAccessor(task=task, data_root=args.data_root)

    exported_train = export_split(
        split_name="source_train",
        source_records=source_train,
        target_records=target_train,
        source_accessor=source_accessor,
        target_accessor=target_accessor,
        output_dir=output_dir,
        image_mode=image_mode,
        image_size=fda_image_size,
        beta=args.beta,
        seed=args.seed,
        skip_existing=args.skip_existing,
    )
    exported_val = export_split(
        split_name="source_val",
        source_records=source_val,
        target_records=target_train,
        source_accessor=source_accessor,
        target_accessor=target_accessor,
        output_dir=output_dir,
        image_mode=image_mode,
        image_size=fda_image_size,
        beta=args.beta,
        seed=args.seed + 1,
        skip_existing=args.skip_existing,
    )

    manifest_dir = os.path.join(output_dir, "manifests")
    ensure_dir(manifest_dir)
    save_json(
        os.path.join(manifest_dir, "source_train.json"),
        {"records": exported_train, "metadata": {"split": "source_train", "beta": args.beta}},
    )
    save_json(
        os.path.join(manifest_dir, "source_val.json"),
        {"records": exported_val, "metadata": {"split": "source_val", "beta": args.beta}},
    )
    save_json(
        os.path.join(output_dir, "export_summary.json"),
        {
            "task": task.name,
            "data_root": os.path.abspath(args.data_root),
            "split_dir": os.path.abspath(split_dir),
            "output_dir": os.path.abspath(output_dir),
            "beta": args.beta,
            "fda_image_size": fda_image_size,
            "image_mode": image_mode,
            "source_train_count": len(exported_train),
            "source_val_count": len(exported_val),
            "target_pool_count": len(target_train),
        },
    )

    print("FDA export finished.")
    print(f"source_train exported: {len(exported_train)}")
    print(f"source_val exported: {len(exported_val)}")


if __name__ == "__main__":
    main()
