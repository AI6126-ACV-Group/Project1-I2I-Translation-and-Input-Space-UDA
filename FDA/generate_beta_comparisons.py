import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageOps

from FDA.data import get_digit_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MNIST -> USPS comparison figures for different FDA beta values."
    )
    parser.add_argument("--data-root", type=str, default="./Datasets")
    parser.add_argument("--bad-export-root", type=str, required=True)
    parser.add_argument("--good-export-root", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./FDA/outputs/figures/mnist2usps_beta_compare",
    )
    parser.add_argument("--split", type=str, default="source_val", choices=["source_train", "source_val"])
    parser.add_argument(
        "--classes",
        type=str,
        default="0,1,2,3",
        help="Comma-separated class ids to visualize, for example: 0,1,2,3",
    )
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--panel-size", type=int, default=224)
    parser.add_argument("--bad-label", type=str, default="FDA beta>0.1")
    parser.add_argument("--good-label", type=str, default="FDA beta=0.071")
    return parser.parse_args()


def load_records(manifest_path: Path) -> List[Dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def parse_ref(ref: str) -> Tuple[str, str, int]:
    dataset_name, split_name, index = ref.split(":")
    return dataset_name, split_name, int(index)


def build_index(records: Sequence[Dict]) -> Dict[str, Dict]:
    return {record["source_reference"]: record for record in records}


def choose_records(records: Sequence[Dict], classes: Sequence[str], samples_per_class: int) -> List[Dict]:
    picked: List[Dict] = []
    counts = {class_name: 0 for class_name in classes}
    for record in records:
        class_name = str(record["class_name"])
        if class_name not in counts:
            continue
        if counts[class_name] >= samples_per_class:
            continue
        picked.append(record)
        counts[class_name] += 1
        if all(count >= samples_per_class for count in counts.values()):
            break
    return picked


def prep(image: Image.Image, panel_size: int) -> Image.Image:
    image = image.convert("L").resize((panel_size, panel_size), Image.Resampling.NEAREST)
    return ImageOps.expand(image, border=2, fill=255)


def draw_sample(
    source_image: Image.Image,
    bad_image: Image.Image,
    good_image: Image.Image,
    target_image: Image.Image,
    titles: Sequence[str],
    panel_size: int,
) -> Image.Image:
    title_height = 28
    gap = 8
    panels = [prep(source_image, panel_size), prep(bad_image, panel_size), prep(good_image, panel_size), prep(target_image, panel_size)]
    canvas_width = len(panels) * panel_size + (len(panels) - 1) * gap
    canvas_height = panel_size + title_height
    canvas = Image.new("L", (canvas_width, canvas_height), color=255)
    draw = ImageDraw.Draw(canvas)

    for index, (panel, title) in enumerate(zip(panels, titles)):
        x = index * (panel_size + gap)
        canvas.paste(panel, (x, title_height))
        draw.text((x + 4, 6), title, fill=0)

    return canvas


def main() -> None:
    args = parse_args()
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]

    bad_root = Path(args.bad_export_root)
    good_root = Path(args.good_export_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bad_records = load_records(bad_root / "manifests" / f"{args.split}.json")
    good_records = load_records(good_root / "manifests" / f"{args.split}.json")
    bad_by_source = build_index(bad_records)
    selected = choose_records(good_records, classes=classes, samples_per_class=args.samples_per_class)

    task_root = str(Path(args.data_root) / "MNIST2USPS")
    mnist_train = get_digit_dataset("mnist", task_root, train=True, transform=None)
    usps_train = get_digit_dataset("usps", task_root, train=True, transform=None)

    rows: List[Image.Image] = []
    selection_payload: List[Dict] = []

    for item_idx, good_record in enumerate(selected, start=1):
        source_ref = good_record["source_reference"]
        if source_ref not in bad_by_source:
            raise KeyError(f"Missing matching source_reference in bad export: {source_ref}")
        bad_record = bad_by_source[source_ref]

        _, _, source_index = parse_ref(good_record["source_reference"])
        _, _, target_index = parse_ref(good_record["target_reference"])

        source_image, source_label = mnist_train[source_index]
        target_image, target_label = usps_train[target_index]
        bad_image = Image.open(bad_root / bad_record["rel_path"]).convert("L")
        good_image = Image.open(good_root / good_record["rel_path"]).convert("L")

        row = draw_sample(
            source_image=source_image,
            bad_image=bad_image,
            good_image=good_image,
            target_image=target_image,
            titles=[
                f"MNIST #{source_index} label={source_label}",
                args.bad_label,
                args.good_label,
                f"USPS ref #{target_index} label={target_label}",
            ],
            panel_size=args.panel_size,
        )
        rows.append(row)

        class_name = str(good_record["class_name"])
        per_sample_path = output_dir / f"{item_idx:02d}_class{class_name}_src{source_index}.png"
        row.save(per_sample_path)

        selection_payload.append(
            {
                "output_path": str(per_sample_path).replace("\\", "/"),
                "class_name": class_name,
                "source_reference": good_record["source_reference"],
                "target_reference": good_record["target_reference"],
                "bad_rel_path": bad_record["rel_path"],
                "good_rel_path": good_record["rel_path"],
            }
        )

    if not rows:
        raise RuntimeError("No samples selected. Check the manifest paths and class ids.")

    gap = 8
    overview_width = max(row.width for row in rows)
    overview_height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    overview = Image.new("L", (overview_width, overview_height), color=255)
    cursor_y = 0
    for row in rows:
        overview.paste(row, (0, cursor_y))
        cursor_y += row.height + gap

    overview_path = output_dir / "overview.png"
    overview.save(overview_path)

    with (output_dir / "selection.json").open("w", encoding="utf-8") as f:
        json.dump(selection_payload, f, indent=2)

    print(f"Saved {len(rows)} comparison rows to: {output_dir}")
    print(f"Overview figure: {overview_path}")


if __name__ == "__main__":
    main()
