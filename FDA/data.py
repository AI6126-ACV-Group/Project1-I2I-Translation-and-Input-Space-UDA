import gzip
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from FDA.common import ensure_dir, load_json, save_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class TaskConfig:
    name: str
    task_type: str
    dataset_subdir: str
    source_domain: str
    target_domain: str
    num_classes: int
    default_image_size: int
    export_image_size: int
    source_val_ratio: float
    target_test_ratio: Optional[float]


TASKS: Dict[str, TaskConfig] = {
    "mnist2usps": TaskConfig(
        name="mnist2usps",
        task_type="digits",
        dataset_subdir="MNIST2USPS",
        source_domain="MNIST",
        target_domain="usps",
        num_classes=10,
        default_image_size=224,
        export_image_size=28,
        source_val_ratio=0.1,
        target_test_ratio=None,
    ),
    "office31_a2w": TaskConfig(
        name="office31_a2w",
        task_type="imagefolder",
        dataset_subdir="office31",
        source_domain="amazon",
        target_domain="webcam",
        num_classes=31,
        default_image_size=224,
        export_image_size=224,
        source_val_ratio=0.1,
        target_test_ratio=0.5,
    ),
    "officehome_a2r": TaskConfig(
        name="officehome_a2r",
        task_type="imagefolder",
        dataset_subdir="OfficeHome",
        source_domain="Art",
        target_domain="RealWorld",
        num_classes=65,
        default_image_size=224,
        export_image_size=224,
        source_val_ratio=0.1,
        target_test_ratio=0.2,
    ),
    "pacs_p2s": TaskConfig(
        name="pacs_p2s",
        task_type="imagefolder",
        dataset_subdir="PACS",
        source_domain="photo",
        target_domain="sketch",
        num_classes=7,
        default_image_size=224,
        export_image_size=224,
        source_val_ratio=0.1,
        target_test_ratio=0.2,
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    if task_name not in TASKS:
        raise ValueError(f"Unsupported task: {task_name}. Choices: {sorted(TASKS)}")
    return TASKS[task_name]


def resolve_task_root(data_root: str, task: TaskConfig) -> str:
    task_root = os.path.join(data_root, task.dataset_subdir)
    if not os.path.isdir(task_root):
        raise FileNotFoundError(f"Task root not found: {task_root}")
    return task_root


def save_manifest(path: str, records: Sequence[Dict], metadata: Optional[Dict] = None) -> None:
    payload = {"records": list(records)}
    if metadata:
        payload["metadata"] = metadata
    save_json(path, payload)


def load_manifest_records(path: str) -> List[Dict]:
    data = load_json(path)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def summarize_records(records: Sequence[Dict]) -> Dict:
    counts = Counter(record["class_name"] for record in records)
    return {
        "num_samples": len(records),
        "per_class": dict(sorted(counts.items())),
    }


def save_split_bundle(
    output_dir: str,
    task: TaskConfig,
    data_root: str,
    source_train: Sequence[Dict],
    source_val: Sequence[Dict],
    target_train: Sequence[Dict],
    target_test: Sequence[Dict],
    class_to_idx: Dict[str, int],
    seed: int,
    source_test: Optional[Sequence[Dict]] = None,
) -> None:
    ensure_dir(output_dir)
    save_manifest(
        os.path.join(output_dir, "source_train.json"),
        source_train,
        metadata={"domain": task.source_domain, "split": "train"},
    )
    save_manifest(
        os.path.join(output_dir, "source_val.json"),
        source_val,
        metadata={"domain": task.source_domain, "split": "val"},
    )
    save_manifest(
        os.path.join(output_dir, "target_train.json"),
        target_train,
        metadata={"domain": task.target_domain, "split": "train_unlabeled"},
    )
    save_manifest(
        os.path.join(output_dir, "target_test.json"),
        target_test,
        metadata={"domain": task.target_domain, "split": "test"},
    )
    if source_test is not None:
        save_manifest(
            os.path.join(output_dir, "source_test.json"),
            source_test,
            metadata={"domain": task.source_domain, "split": "test"},
        )

    save_json(
        os.path.join(output_dir, "classes.json"),
        {
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(idx): class_name for class_name, idx in class_to_idx.items()},
        },
    )
    save_json(
        os.path.join(output_dir, "meta.json"),
        {
            "task": task.name,
            "task_type": task.task_type,
            "data_root": os.path.abspath(data_root),
            "task_root": os.path.abspath(resolve_task_root(data_root, task)),
            "source_domain": task.source_domain,
            "target_domain": task.target_domain,
            "num_classes": task.num_classes,
            "source_val_ratio": task.source_val_ratio,
            "target_test_ratio": task.target_test_ratio,
            "seed": seed,
        },
    )
    stats_payload = {
        "source_train": summarize_records(source_train),
        "source_val": summarize_records(source_val),
        "target_train": summarize_records(target_train),
        "target_test": summarize_records(target_test),
    }
    if source_test is not None:
        stats_payload["source_test"] = summarize_records(source_test)
    save_json(os.path.join(output_dir, "stats.json"), stats_payload)


def stratified_split(records: Sequence[Dict], ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not 0.0 < ratio < 1.0:
        raise ValueError("Split ratio must be between 0 and 1.")

    grouped: Dict[int, List[Dict]] = {}
    for record in records:
        grouped.setdefault(record["class_idx"], []).append(dict(record))

    rng = random.Random(seed)
    first: List[Dict] = []
    second: List[Dict] = []
    for class_idx in sorted(grouped):
        items = list(grouped[class_idx])
        rng.shuffle(items)
        n_items = len(items)
        if n_items < 2:
            first.extend(items)
            continue
        n_second = max(1, min(n_items - 1, int(round(n_items * ratio))))
        second.extend(items[:n_second])
        first.extend(items[n_second:])

    first.sort(key=lambda item: (item["class_idx"], item.get("rel_path", ""), item.get("index", -1)))
    second.sort(key=lambda item: (item["class_idx"], item.get("rel_path", ""), item.get("index", -1)))
    return first, second


def _find_case_insensitive_subdir(root: str, expected: str) -> str:
    direct = os.path.join(root, expected)
    if os.path.isdir(direct):
        return direct

    expected_lower = expected.lower()
    candidates = [
        entry
        for entry in os.listdir(root)
        if os.path.isdir(os.path.join(root, entry)) and entry.lower() == expected_lower
    ]
    if len(candidates) == 1:
        return os.path.join(root, candidates[0])

    raise FileNotFoundError(f"Unable to find subdirectory '{expected}' under: {root}")


def resolve_domain_image_root(task_root: str, domain: str) -> str:
    domain_root = _find_case_insensitive_subdir(task_root, domain)
    candidates = [
        os.path.join(domain_root, "images"),
        domain_root,
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            class_dirs = [
                entry for entry in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, entry))
            ]
            if class_dirs:
                return candidate
    raise FileNotFoundError(
        f"Unable to locate image root for domain '{domain}' under task root: {task_root}"
    )


def _is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def discover_domain_samples(task_root: str, domain: str) -> Tuple[List[Dict], Dict[str, int]]:
    image_root = resolve_domain_image_root(task_root, domain)
    class_names = sorted(
        entry
        for entry in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, entry))
    )
    if not class_names:
        raise RuntimeError(f"No class folders found in: {image_root}")

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    records: List[Dict] = []
    sample_idx = 0
    for class_name in class_names:
        class_dir = os.path.join(image_root, class_name)
        for filename in sorted(os.listdir(class_dir)):
            if not _is_image_file(filename):
                continue
            abs_path = os.path.join(class_dir, filename)
            rel_path = os.path.relpath(abs_path, task_root).replace("\\", "/")
            records.append(
                {
                    "sample_idx": sample_idx,
                    "domain": domain,
                    "rel_path": rel_path,
                    "class_name": class_name,
                    "class_idx": class_to_idx[class_name],
                    "filename": filename,
                }
            )
            sample_idx += 1

    if not records:
        raise RuntimeError(f"No images found for domain '{domain}' in: {image_root}")
    return records, class_to_idx


def build_imagefolder_splits(
    task: TaskConfig,
    data_root: str,
    output_dir: str,
    source_val_ratio: Optional[float] = None,
    target_test_ratio: Optional[float] = None,
    seed: int = 42,
) -> None:
    task_root = resolve_task_root(data_root, task)
    source_records, source_class_to_idx = discover_domain_samples(task_root, task.source_domain)
    target_records, target_class_to_idx = discover_domain_samples(task_root, task.target_domain)

    if source_class_to_idx != target_class_to_idx:
        raise ValueError("Source and target class mappings do not match.")

    source_train, source_val = stratified_split(
        source_records, ratio=source_val_ratio or task.source_val_ratio, seed=seed
    )
    target_train, target_test = stratified_split(
        target_records, ratio=target_test_ratio or task.target_test_ratio, seed=seed + 1
    )

    save_split_bundle(
        output_dir=output_dir,
        task=task,
        data_root=data_root,
        source_train=source_train,
        source_val=source_val,
        target_train=target_train,
        target_test=target_test,
        class_to_idx=source_class_to_idx,
        seed=seed,
    )


def _read_idx_array(path: str) -> np.ndarray:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        buffer = f.read()

    magic = int.from_bytes(buffer[0:4], byteorder="big")
    dims = magic & 0xFF
    dtype_code = (magic >> 8) & 0xFF
    if dtype_code != 0x08:
        raise ValueError(f"Unsupported IDX dtype code in {path}: {dtype_code}")

    shape = []
    offset = 4
    for _ in range(dims):
        shape.append(int.from_bytes(buffer[offset : offset + 4], byteorder="big"))
        offset += 4

    array = np.frombuffer(buffer, dtype=np.uint8, offset=offset)
    return array.reshape(shape)


def _find_existing_path(directory: str, filename: str) -> Optional[str]:
    plain_path = os.path.join(directory, filename)
    gz_path = plain_path + ".gz"
    if os.path.exists(plain_path):
        return plain_path
    if os.path.exists(gz_path):
        return gz_path
    return None


def _locate_mnist_files(task_root: str, train: bool) -> Optional[Tuple[str, str]]:
    image_filename = "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
    label_filename = "train-labels-idx1-ubyte" if train else "t10k-labels-idx1-ubyte"

    dataset_root = os.path.join(task_root, "MNIST")
    candidate_dirs = [
        dataset_root,
        os.path.join(dataset_root, "raw"),
        os.path.join(dataset_root, "MNIST"),
        os.path.join(dataset_root, "MNIST", "raw"),
        task_root,
        os.path.join(task_root, "raw"),
    ]

    seen = set()
    for directory in candidate_dirs:
        normalized = os.path.normpath(directory)
        if normalized in seen or not os.path.isdir(directory):
            continue
        seen.add(normalized)

        image_path = _find_existing_path(directory, image_filename)
        label_path = _find_existing_path(directory, label_filename)
        if image_path and label_path:
            return image_path, label_path
    return None


class LocalMNISTDataset(Dataset):
    def __init__(self, image_path: str, label_path: str, transform=None):
        self.transform = transform
        self.images = _read_idx_array(image_path)
        self.labels = _read_idx_array(label_path)

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"MNIST image/label count mismatch: {len(self.images)} vs {len(self.labels)}"
            )

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index], mode="L")
        label = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        return len(self.labels)


def get_digit_dataset(dataset_name: str, task_root: str, train: bool, transform=None) -> Dataset:
    dataset_name = dataset_name.lower()
    dataset_root = os.path.join(task_root, dataset_name)

    if dataset_name == "mnist":
        local_files = _locate_mnist_files(task_root, train=train)
        if local_files is not None:
            image_path, label_path = local_files
            return LocalMNISTDataset(image_path=image_path, label_path=label_path, transform=transform)
        try:
            return datasets.MNIST(dataset_root, train=train, transform=transform, download=False)
        except RuntimeError:
            return datasets.MNIST(dataset_root, train=train, transform=transform, download=True)

    if dataset_name == "usps":
        try:
            return datasets.USPS(dataset_root, train=train, transform=transform, download=False)
        except (RuntimeError, EOFError, OSError):
            return datasets.USPS(dataset_root, train=train, transform=transform, download=True)

    raise ValueError(f"Unsupported digit dataset: {dataset_name}")


def _extract_targets(dataset: Dataset) -> List[int]:
    for attr_name in ("targets", "labels"):
        if hasattr(dataset, attr_name):
            values = getattr(dataset, attr_name)
            if isinstance(values, torch.Tensor):
                return values.tolist()
            return [int(item) for item in values]

    labels = []
    for index in range(len(dataset)):
        _, label = dataset[index]
        labels.append(int(label))
    return labels


def build_digit_records(dataset_name: str, split_name: str, labels: Sequence[int]) -> List[Dict]:
    records = []
    for index, label in enumerate(labels):
        records.append(
            {
                "dataset": dataset_name.lower(),
                "split": split_name,
                "index": index,
                "class_idx": int(label),
                "class_name": str(int(label)),
                "filename": f"{index:06d}.png",
            }
        )
    return records


def build_digit_splits(
    task: TaskConfig,
    data_root: str,
    output_dir: str,
    source_val_ratio: Optional[float] = None,
    seed: int = 42,
) -> None:
    task_root = resolve_task_root(data_root, task)
    source_train_full = get_digit_dataset(task.source_domain, task_root, train=True, transform=None)
    source_test_ds = get_digit_dataset(task.source_domain, task_root, train=False, transform=None)
    target_train_ds = get_digit_dataset(task.target_domain, task_root, train=True, transform=None)
    target_test_ds = get_digit_dataset(task.target_domain, task_root, train=False, transform=None)

    source_train_records = build_digit_records(
        task.source_domain, "train", _extract_targets(source_train_full)
    )
    source_test_records = build_digit_records(
        task.source_domain, "test", _extract_targets(source_test_ds)
    )
    target_train_records = build_digit_records(
        task.target_domain, "train", _extract_targets(target_train_ds)
    )
    target_test_records = build_digit_records(
        task.target_domain, "test", _extract_targets(target_test_ds)
    )

    source_train, source_val = stratified_split(
        source_train_records, ratio=source_val_ratio or task.source_val_ratio, seed=seed
    )
    class_to_idx = {str(idx): idx for idx in range(task.num_classes)}

    save_split_bundle(
        output_dir=output_dir,
        task=task,
        data_root=data_root,
        source_train=source_train,
        source_val=source_val,
        target_train=target_train_records,
        target_test=target_test_records,
        class_to_idx=class_to_idx,
        seed=seed,
        source_test=source_test_records,
    )


class RecordAccessor:
    def __init__(self, task: TaskConfig, data_root: str):
        self.task = task
        self.data_root = data_root
        self.task_root = resolve_task_root(data_root, task)
        self._digit_cache: Dict[Tuple[str, str], Dataset] = {}

    def load_image(self, record: Dict) -> Image.Image:
        if self.task.task_type == "imagefolder":
            path = os.path.join(self.task_root, record["rel_path"])
            image = Image.open(path).convert("RGB")
            return ImageOps.exif_transpose(image)

        dataset_name = record["dataset"]
        split_name = record["split"]
        key = (dataset_name, split_name)
        if key not in self._digit_cache:
            self._digit_cache[key] = get_digit_dataset(
                dataset_name,
                self.task_root,
                train=(split_name == "train"),
                transform=None,
            )
        image, _ = self._digit_cache[key][record["index"]]
        return image.convert("L")


class RawManifestDataset(Dataset):
    def __init__(
        self,
        task: TaskConfig,
        data_root: str,
        records: Sequence[Dict],
        transform,
    ):
        self.records = list(records)
        self.transform = transform
        self.accessor = RecordAccessor(task=task, data_root=data_root)
        if not self.records:
            raise RuntimeError("Dataset is empty.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = self.accessor.load_image(record)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(record["class_idx"])


class ExportedManifestImageDataset(Dataset):
    def __init__(self, root: str, records: Sequence[Dict], transform):
        self.root = root
        self.records = list(records)
        self.transform = transform
        if not self.records:
            raise RuntimeError("Dataset is empty.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = os.path.join(self.root, record["rel_path"])
        mode = record.get("image_mode", "RGB")
        image = Image.open(path).convert(mode)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(record["class_idx"])


def build_train_transform(
    task: TaskConfig,
    image_size: int,
    pretrained: bool,
    augment: str = "default",
):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    if task.task_type == "digits":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    if augment == "resize_flip":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transform(task: TaskConfig, image_size: int, pretrained: bool):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    ops = [transforms.Resize((image_size, image_size))]
    if task.task_type == "digits":
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


def build_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def build_fda_training_loaders(
    task: TaskConfig,
    data_root: str,
    split_dir: str,
    fda_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pretrained: bool,
    train_augment: str = "default",
) -> Dict[str, DataLoader]:
    train_transform = build_train_transform(
        task,
        image_size=image_size,
        pretrained=pretrained,
        augment=train_augment,
    )
    eval_transform = build_eval_transform(task, image_size=image_size, pretrained=pretrained)

    fda_manifest_dir = os.path.join(fda_root, "manifests")
    fda_train_records = load_manifest_records(os.path.join(fda_manifest_dir, "source_train.json"))
    fda_val_records = load_manifest_records(os.path.join(fda_manifest_dir, "source_val.json"))
    target_test_records = load_manifest_records(os.path.join(split_dir, "target_test.json"))

    loaders = {
        "fda_train": build_loader(
            ExportedManifestImageDataset(fda_root, fda_train_records, transform=train_transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        ),
        "fda_val": build_loader(
            ExportedManifestImageDataset(fda_root, fda_val_records, transform=eval_transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
        "target_test": build_loader(
            RawManifestDataset(task, data_root, target_test_records, transform=eval_transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
    }

    source_test_path = os.path.join(split_dir, "source_test.json")
    if os.path.exists(source_test_path):
        source_test_records = load_manifest_records(source_test_path)
        loaders["source_test"] = build_loader(
            RawManifestDataset(task, data_root, source_test_records, transform=eval_transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    return loaders
