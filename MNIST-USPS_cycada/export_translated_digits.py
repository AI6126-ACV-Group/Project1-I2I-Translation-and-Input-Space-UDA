import argparse
import os

import torch

from digit_uda.cyclegan import ResNetGenerator, build_unpaired_digit_loaders, export_translated_digits
from digit_uda.common import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export translated digit images from a trained CycleGAN generator.")
    parser.add_argument("--source", type=str, default="mnist", choices=["mnist", "usps"])
    parser.add_argument("--target", type=str, default="usps", choices=["mnist", "usps"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/translated_digits/mnist_to_usps")
    parser.add_argument("--zip-output", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_project_path(path: str, project_root: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(project_root, path))


def resolve_data_root(data_root: str, project_root: str, source: str, target: str) -> str:
    requested_root = resolve_project_path(data_root, project_root)
    project_data_root = os.path.join(project_root, "data")
    requested_ok = os.path.isdir(os.path.join(requested_root, source)) or os.path.isdir(
        os.path.join(requested_root, target)
    )
    project_ok = os.path.isdir(os.path.join(project_data_root, source)) or os.path.isdir(
        os.path.join(project_data_root, target)
    )
    if requested_ok:
        return requested_root
    if project_ok:
        print(f"Using project-local data root instead of missing dataset path: {project_data_root}")
        return project_data_root
    return requested_root


def main() -> None:
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    args.data_root = resolve_data_root(args.data_root, project_root, args.source, args.target)
    args.checkpoint = resolve_project_path(args.checkpoint, project_root)
    args.output_dir = resolve_project_path(args.output_dir, project_root)
    args.zip_output = resolve_project_path(args.zip_output, project_root) if args.zip_output else ""

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_unpaired_digit_loaders(
        source=args.source,
        target=args.target,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    export_loader = loaders["source_export"]

    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator = ResNetGenerator().to(device)
    if "modules" in checkpoint and "G_A2B" in checkpoint["modules"]:
        generator.load_state_dict(checkpoint["modules"]["G_A2B"])
    else:
        generator.load_state_dict(checkpoint)

    max_items = args.limit if args.limit > 0 else None
    result = export_translated_digits(
        generator=generator,
        loader=export_loader,
        output_dir=args.output_dir,
        device=device,
        max_items=max_items,
        zip_output=args.zip_output,
    )

    save_json(os.path.join(args.output_dir, "export_summary.json"), {"args": vars(args), "result": result})
    print(f"Export finished. Saved {result['num_exported']} translated images to {args.output_dir}")
    if args.zip_output:
        print(f"Zip archive: {args.zip_output}")


if __name__ == "__main__":
    main()
