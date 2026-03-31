import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def split_uda_dataset(source_a, source_b, output_root, split_ratio=0.8, extensions=None):
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    subfolders = ['trainA', 'testA', 'trainB', 'testB']
    for folder in subfolders:
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)

    def process_domain(source_dir, domain_label):
        print(f"\nscanning Domain {domain_label} files in {source_dir}...")
        all_files = []

        for root, _, files in os.walk(source_dir):
            for f in files:
                if Path(f).suffix.lower() in extensions:
                    all_files.append(os.path.join(root, f))

        if not all_files:
            print(f" {source_dir} does not contain any {extensions} files.")
            return

        random.shuffle(all_files)

        split_idx = int(len(all_files) * split_ratio)
        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]

        tasks = [
            (train_files, 'train'),
            (test_files, 'test')
        ]

        for files, split_type in tasks:
            target_dir = os.path.join(output_root, f"{split_type}{domain_label}")
            desc = f"Domain {domain_label} [{split_type}]"

            for fpath in tqdm(files, desc=desc, unit="img"):
                rel_path = os.path.relpath(fpath, source_dir)
                new_name = rel_path.replace(os.sep, '_')

                dest_path = os.path.join(target_dir, new_name)
                shutil.copy2(fpath, dest_path)

    process_domain(source_a, 'A')
    process_domain(source_b, 'B')

    print("-" * 30)
    print(f"output dataset path: {os.path.abspath(output_root)}")


if __name__ == "__main__":
    # 请在此处填入你的实际路径
    config = {
        "source_a": "./MINST",
        "source_b": "./USPS",
        "output_root": "MINST2USPS",
        "split_ratio": 0.8
    }

    split_uda_dataset(**config)