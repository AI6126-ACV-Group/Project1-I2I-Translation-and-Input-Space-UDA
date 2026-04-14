import argparse
import os
import random
import shutil
from typing import Dict, List

from PIL import Image, ImageDraw

from office31_uda.common import ensure_dir, load_json, save_json, save_manifest, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CycleGAN fake_B images into amazon2webcam ImageFolder layout."
    )
    parser.add_argument("--cycle-index", type=str, required=True, help="metadata/testA_index.json")
    parser.add_argument(
        "--cyclegan-images-dir",
        type=str,
        required=True,
        help="CycleGAN output image directory containing *_fake_B.png.",
    )
    parser.add_argument("--translated-root", type=str, required=True)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def strip_fake_suffix(filename: str) -> str:
    for suffix in ["_fake_B.png", "_fake_B.jpg", "_fake_B.jpeg"]:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    raise ValueError(f"Unexpected translated filename: {filename}")


def build_preview(output_path: str, preview_pairs: List[Dict]) -> None:
    if not preview_pairs:
        return
    cols = 3
    rows = len(preview_pairs)
    image_size = (256, 256)
    canvas = Image.new("RGB", (cols * image_size[0], rows * image_size[1]), color=(255, 255, 255))

    for row, pair in enumerate(preview_pairs):
        for col, key in enumerate(["source_path", "translated_path", "restored_path"]):
            image = Image.open(pair[key]).convert("RGB").resize(image_size)
            canvas.paste(image, (col * image_size[0], row * image_size[1]))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "source | translated | copied_preview", fill=(255, 0, 0))
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.translated_root)
    manifest_dir = os.path.join(args.translated_root, "manifests")
    image_root = os.path.join(args.translated_root, "images")
    preview_dir = os.path.join(args.translated_root, "previews")
    ensure_dir(manifest_dir)
    ensure_dir(image_root)
    ensure_dir(preview_dir)

    index_payload = load_json(args.cycle_index)
    records = index_payload["records"]
    data_root = index_payload["data_root"]
    by_staged_key = {}
    for record in records:
        # CycleGAN fake_B filenames are based on the staged stem and usually do
        # not preserve the original source extension, so support both keys.
        by_staged_key[record["staged_name"]] = record
        by_staged_key[record["staged_stem"]] = record

    translated_records: List[Dict] = []
    split_groups: Dict[str, List[Dict]] = {}
    preview_pairs: List[Dict] = []
    translated_files = sorted(
        filename
        for filename in os.listdir(args.cyclegan_images_dir)
        if "_fake_B" in filename
    )
    if not translated_files:
        raise RuntimeError(f"No *_fake_B files found in: {args.cyclegan_images_dir}")

    for filename in translated_files:
        staged_name = strip_fake_suffix(filename)
        if staged_name not in by_staged_key:
            continue

        record = by_staged_key[staged_name]
        src_file = os.path.join(args.cyclegan_images_dir, filename)
        class_dir = os.path.join(image_root, record["class_name"])
        ensure_dir(class_dir)

        dst_filename = f"{record['staged_stem']}_fake_B.png"
        dst_path = os.path.join(class_dir, dst_filename)
        shutil.copy2(src_file, dst_path)

        translated_record = {
            "domain": "amazon2webcam",
            "class_name": record["class_name"],
            "class_idx": record["class_idx"],
            "rel_path": os.path.relpath(dst_path, args.translated_root).replace("\\", "/"),
            "source_rel_path": record["rel_path"],
            "split_name": record["split_name"],
            "filename": dst_filename,
        }
        translated_records.append(translated_record)
        split_groups.setdefault(record["split_name"], []).append(translated_record)

        if len(preview_pairs) < args.preview_count:
            preview_pairs.append(
                {
                    "source_path": os.path.join(data_root, record["rel_path"]),
                    "translated_path": dst_path,
                    "restored_path": dst_path,
                }
            )

    if not translated_records:
        raise RuntimeError("No translated images matched testA index.")

    translated_records.sort(key=lambda item: item["rel_path"])
    save_manifest(
        os.path.join(manifest_dir, "all.json"),
        translated_records,
        metadata={"num_samples": len(translated_records)},
    )
    for split_name, split_records in split_groups.items():
        split_records.sort(key=lambda item: item["rel_path"])
        save_manifest(
            os.path.join(manifest_dir, f"{split_name}.json"),
            split_records,
            metadata={"num_samples": len(split_records), "split_name": split_name},
        )

    build_preview(os.path.join(preview_dir, "translation_preview.png"), preview_pairs)
    save_json(
        os.path.join(args.translated_root, "export_meta.json"),
        {
            "cycle_index": os.path.abspath(args.cycle_index),
            "cyclegan_images_dir": os.path.abspath(args.cyclegan_images_dir),
            "num_exported": len(translated_records),
        },
    )

    print("Exported translated dataset to:", os.path.abspath(args.translated_root))
    print("num_exported:", len(translated_records))


if __name__ == "__main__":
    main()
