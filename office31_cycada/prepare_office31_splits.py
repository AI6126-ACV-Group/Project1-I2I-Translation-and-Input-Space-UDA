import argparse
import os
from collections import Counter

from office31_uda.common import (
    discover_domain_samples,
    ensure_dir,
    save_json,
    save_manifest,
    set_seed,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible Office-31 manifests for Amazon -> Webcam."
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./splits/amazon_to_webcam")
    parser.add_argument("--source-domain", type=str, default="amazon")
    parser.add_argument("--target-domain", type=str, default="webcam")
    parser.add_argument("--source-val-ratio", type=float, default=0.1)
    parser.add_argument("--target-test-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def summarize(records):
    counts = Counter(record["class_name"] for record in records)
    return {
        "num_samples": len(records),
        "per_class": dict(sorted(counts.items())),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    source_records, source_class_to_idx = discover_domain_samples(args.data_root, args.source_domain)
    target_records, target_class_to_idx = discover_domain_samples(args.data_root, args.target_domain)

    if source_class_to_idx != target_class_to_idx:
        raise ValueError("Source and target class mappings do not match.")

    amazon_train, amazon_val = stratified_split(
        source_records, ratio=args.source_val_ratio, seed=args.seed
    )
    webcam_train, webcam_test = stratified_split(
        target_records, ratio=args.target_test_ratio, seed=args.seed + 1
    )

    save_manifest(
        os.path.join(args.output_dir, "amazon_train.json"),
        amazon_train,
        metadata={"domain": args.source_domain, "split": "train"},
    )
    save_manifest(
        os.path.join(args.output_dir, "amazon_val.json"),
        amazon_val,
        metadata={"domain": args.source_domain, "split": "val"},
    )
    save_manifest(
        os.path.join(args.output_dir, "webcam_train.json"),
        webcam_train,
        metadata={"domain": args.target_domain, "split": "train_unlabeled"},
    )
    save_manifest(
        os.path.join(args.output_dir, "webcam_test.json"),
        webcam_test,
        metadata={"domain": args.target_domain, "split": "test"},
    )

    save_json(
        os.path.join(args.output_dir, "classes.json"),
        {
            "class_to_idx": source_class_to_idx,
            "idx_to_class": {str(v): k for k, v in source_class_to_idx.items()},
        },
    )
    save_json(
        os.path.join(args.output_dir, "meta.json"),
        {
            "data_root": os.path.abspath(args.data_root),
            "source_domain": args.source_domain,
            "target_domain": args.target_domain,
            "source_val_ratio": args.source_val_ratio,
            "target_test_ratio": args.target_test_ratio,
            "seed": args.seed,
        },
    )
    save_json(
        os.path.join(args.output_dir, "stats.json"),
        {
            "amazon_train": summarize(amazon_train),
            "amazon_val": summarize(amazon_val),
            "webcam_train": summarize(webcam_train),
            "webcam_test": summarize(webcam_test),
        },
    )

    print("Saved split manifests to:", os.path.abspath(args.output_dir))
    print("amazon_train:", len(amazon_train))
    print("amazon_val:", len(amazon_val))
    print("webcam_train_unlabeled:", len(webcam_train))
    print("webcam_test:", len(webcam_test))


if __name__ == "__main__":
    main()
