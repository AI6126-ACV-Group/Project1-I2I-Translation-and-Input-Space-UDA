import json
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


NUM_CLASSES = 31
VALID_DOMAINS = ("amazon", "dslr", "webcam")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_logging(output_dir: str, filename: str = "train.log"):
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, filename)
    file_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, file_handle)
    sys.stderr = TeeStream(sys.__stderr__, file_handle)
    return file_handle


def resolve_domain_image_root(data_root: str, domain: str) -> str:
    candidates = [
        os.path.join(data_root, domain, "images"),
        os.path.join(data_root, domain),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Unable to locate image root for domain '{domain}' under data root: {data_root}"
    )


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def discover_domain_samples(data_root: str, domain: str) -> Tuple[List[Dict], Dict[str, int]]:
    image_root = resolve_domain_image_root(data_root, domain)
    class_names = sorted(
        entry
        for entry in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, entry))
    )
    if not class_names:
        raise RuntimeError(f"No class folders found in: {image_root}")

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    records: List[Dict] = []
    for class_name in class_names:
        class_dir = os.path.join(image_root, class_name)
        for filename in sorted(os.listdir(class_dir)):
            if not is_image_file(filename):
                continue
            abs_path = os.path.join(class_dir, filename)
            rel_path = os.path.relpath(abs_path, data_root).replace("\\", "/")
            records.append(
                {
                    "domain": domain,
                    "rel_path": rel_path,
                    "class_name": class_name,
                    "class_idx": class_to_idx[class_name],
                    "filename": filename,
                }
            )

    if not records:
        raise RuntimeError(f"No images found for domain '{domain}' in: {image_root}")
    return records, class_to_idx


def stratified_split(records: Sequence[Dict], ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not 0.0 < ratio < 1.0:
        raise ValueError("Split ratio must be between 0 and 1.")

    grouped: Dict[int, List[Dict]] = {}
    for record in records:
        grouped.setdefault(record["class_idx"], []).append(record)

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

    first.sort(key=lambda item: item["rel_path"])
    second.sort(key=lambda item: item["rel_path"])
    return first, second


def make_cycle_stem(index: int, record: Dict) -> str:
    base_name = os.path.splitext(record["filename"])[0]
    safe_class = record["class_name"].replace(" ", "_")
    return f"{index:06d}__{safe_class}__{base_name}"


def load_manifest_records(path: str) -> List[Dict]:
    data = load_json(path)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def save_manifest(path: str, records: Sequence[Dict], metadata: Optional[Dict] = None) -> None:
    payload = {"records": list(records)}
    if metadata:
        payload["metadata"] = metadata
    save_json(path, payload)


def build_train_transform(image_size: int, pretrained: bool) -> transforms.Compose:
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transform(image_size: int, pretrained: bool) -> transforms.Compose:
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict],
        data_root: str,
        transform: transforms.Compose,
        allow_missing_labels: bool = False,
    ):
        self.records = list(records)
        self.data_root = data_root
        self.transform = transform
        self.allow_missing_labels = allow_missing_labels
        if not self.records:
            raise RuntimeError("Dataset is empty.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = os.path.join(self.data_root, record["rel_path"])
        image = Image.open(path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        if self.transform is not None:
            image = self.transform(image)
        label = record.get("class_idx", -1)
        if label == -1 and not self.allow_missing_labels:
            raise ValueError(f"Missing class_idx for record: {record}")
        return image, int(label)


def build_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def build_source_only_loaders(
    data_root: str,
    split_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pretrained: bool,
) -> Dict[str, DataLoader]:
    train_transform = build_train_transform(image_size=image_size, pretrained=pretrained)
    eval_transform = build_eval_transform(image_size=image_size, pretrained=pretrained)

    amazon_train = load_manifest_records(os.path.join(split_dir, "amazon_train.json"))
    amazon_val = load_manifest_records(os.path.join(split_dir, "amazon_val.json"))
    webcam_test = load_manifest_records(os.path.join(split_dir, "webcam_test.json"))

    return {
        "src_train": build_loader(
            ManifestImageDataset(amazon_train, data_root=data_root, transform=train_transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        ),
        "src_val": build_loader(
            ManifestImageDataset(amazon_val, data_root=data_root, transform=eval_transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
        "tgt_test": build_loader(
            ManifestImageDataset(webcam_test, data_root=data_root, transform=eval_transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
    }


def build_translated_loaders(
    translated_root: str,
    split_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pretrained: bool,
) -> Dict[str, DataLoader]:
    train_transform = build_train_transform(image_size=image_size, pretrained=pretrained)
    eval_transform = build_eval_transform(image_size=image_size, pretrained=pretrained)

    translated_manifest_dir = os.path.join(translated_root, "manifests")
    translated_train = load_manifest_records(os.path.join(translated_manifest_dir, "amazon_train.json"))
    translated_val = load_manifest_records(os.path.join(translated_manifest_dir, "amazon_val.json"))
    webcam_train = load_manifest_records(os.path.join(split_dir, "webcam_train.json"))
    webcam_test = load_manifest_records(os.path.join(split_dir, "webcam_test.json"))

    return {
        "translated_train": build_loader(
            ManifestImageDataset(translated_train, data_root=translated_root, transform=train_transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        ),
        "translated_val": build_loader(
            ManifestImageDataset(translated_val, data_root=translated_root, transform=eval_transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
        "target_train": build_loader(
            ManifestImageDataset(webcam_train, data_root=data_root_for_split(split_dir), transform=train_transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        ),
        "target_test": build_loader(
            ManifestImageDataset(webcam_test, data_root=data_root_for_split(split_dir), transform=eval_transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
    }


def data_root_for_split(split_dir: str) -> str:
    meta_path = os.path.join(split_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing split metadata: {meta_path}")
    meta = load_json(meta_path)
    return meta["data_root"]


class ResNet18FeatureNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = False):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except AttributeError:
            backbone = models.resnet18(pretrained=pretrained)

        in_features = backbone.fc.in_features
        self.encoder = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
        self.classifier = nn.Linear(in_features, num_classes)
        self.feature_dim = in_features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.encoder(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForeverDataIterator:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def create_writer(output_dir: str):
    if SummaryWriter is None:
        return None
    log_dir = os.path.join(output_dir, "tensorboard")
    ensure_dir(log_dir)
    return SummaryWriter(log_dir=log_dir)


def log_scalars(writer, metrics: Dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, step)


def save_checkpoint(
    path: str,
    modules: Dict[str, nn.Module],
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    schedulers: Optional[Dict[str, object]] = None,
    state: Optional[Dict] = None,
) -> None:
    ensure_dir(os.path.dirname(path))
    checkpoint = {
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "optimizers": {name: opt.state_dict() for name, opt in (optimizers or {}).items()},
        "schedulers": {name: sch.state_dict() for name, sch in (schedulers or {}).items()},
        "state": state or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    modules: Optional[Dict[str, nn.Module]] = None,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    schedulers: Optional[Dict[str, object]] = None,
    map_location: str = "cpu",
) -> Dict:
    checkpoint = torch.load(path, map_location=map_location)
    for name, module in (modules or {}).items():
        module.load_state_dict(checkpoint["modules"][name])
    for name, optimizer in (optimizers or {}).items():
        if name in checkpoint.get("optimizers", {}):
            optimizer.load_state_dict(checkpoint["optimizers"][name])
    for name, scheduler in (schedulers or {}).items():
        if name in checkpoint.get("schedulers", {}):
            scheduler.load_state_dict(checkpoint["schedulers"][name])
    return checkpoint


def plot_history(history: List[Dict], output_dir: str, groups: Dict[str, Iterable[str]]) -> None:
    if plt is None or not history:
        return

    epochs = [item["epoch"] for item in history]
    for filename, keys in groups.items():
        plt.figure(figsize=(8, 5))
        plotted = False
        for key in keys:
            values = [item[key] for item in history if key in item]
            if len(values) == len(epochs):
                plt.plot(epochs, values, label=key)
                plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("Epoch")
        plt.ylabel(filename)
        plt.title(filename)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        ensure_dir(output_dir)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.close()


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def save_progress(output_dir: str, history: List[Dict], summary: Dict) -> None:
    save_json(
        os.path.join(output_dir, "progress.json"),
        {
            "history": history,
            "summary": summary,
        },
    )
