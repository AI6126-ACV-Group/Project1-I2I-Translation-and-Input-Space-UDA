import io
import gzip
import json
import os
import random
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


NUM_CLASSES = 10
SUPPORTED_DATASETS = {"mnist", "usps"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int, pretrained: bool) -> transforms.Compose:
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
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


def _locate_mnist_files(data_root: str, dataset_root: str, train: bool) -> Optional[Tuple[str, str]]:
    image_filename = "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
    label_filename = "train-labels-idx1-ubyte" if train else "t10k-labels-idx1-ubyte"

    candidate_dirs = [
        dataset_root,
        os.path.join(dataset_root, "raw"),
        os.path.join(dataset_root, "MNIST"),
        os.path.join(dataset_root, "MNIST", "raw"),
        data_root,
        os.path.join(data_root, "raw"),
        os.path.join(data_root, "MNIST"),
        os.path.join(data_root, "MNIST", "raw"),
        os.path.join(data_root, "mnist"),
        os.path.join(data_root, "mnist", "raw"),
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
    def __init__(self, image_path: str, label_path: str, transform: transforms.Compose):
        self.image_path = image_path
        self.label_path = label_path
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


def get_digit_dataset(name: str, root: str, train: bool, transform: transforms.Compose) -> Dataset:
    name = name.lower()
    dataset_root = os.path.join(root, name)

    if name == "mnist":
        local_files = _locate_mnist_files(root, dataset_root, train=train)
        if local_files is not None:
            image_path, label_path = local_files
            print(f"Using local MNIST files: {image_path} | {label_path}")
            return LocalMNISTDataset(image_path, label_path, transform=transform)

        try:
            return datasets.MNIST(dataset_root, train=train, transform=transform, download=False)
        except RuntimeError:
            return datasets.MNIST(dataset_root, train=train, transform=transform, download=True)
    if name == "usps":
        try:
            return datasets.USPS(dataset_root, train=train, transform=transform, download=False)
        except (RuntimeError, EOFError, OSError) as exc:
            print(f"Local USPS dataset unavailable or corrupted at {dataset_root}: {exc}")
            return datasets.USPS(dataset_root, train=train, transform=transform, download=True)
    raise ValueError(f"Unsupported dataset: {name}")


def split_train_val(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    val_size = int(len(dataset) * val_ratio)
    if val_size <= 0 or val_size >= len(dataset):
        raise ValueError("val_ratio produces an invalid validation split.")

    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )


def build_source_only_loaders(
    source: str,
    target: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    pretrained: bool,
    seed: int,
) -> Dict[str, DataLoader]:
    transform = build_transforms(image_size=image_size, pretrained=pretrained)
    src_train_full = get_digit_dataset(source, data_root, train=True, transform=transform)
    src_train, src_val = split_train_val(src_train_full, val_ratio=val_ratio, seed=seed)
    src_test = get_digit_dataset(source, data_root, train=False, transform=transform)
    tgt_test = get_digit_dataset(target, data_root, train=False, transform=transform)

    return {
        "src_train": build_loader(src_train, batch_size, num_workers, shuffle=True),
        "src_val": build_loader(src_val, batch_size, num_workers, shuffle=False),
        "src_test": build_loader(src_test, batch_size, num_workers, shuffle=False),
        "tgt_test": build_loader(tgt_test, batch_size, num_workers, shuffle=False),
    }


class TranslatedDigitsDataset(Dataset):
    def __init__(self, root_or_zip: str, transform: transforms.Compose, suffix: str = "_fake_B.png"):
        self.root_or_zip = root_or_zip
        self.transform = transform
        self.suffix = suffix
        self.is_zip = zipfile.is_zipfile(root_or_zip)
        self._zip_file = None
        self.samples = self._scan_samples()
        if not self.samples:
            raise RuntimeError(f"No translated images matching '*{suffix}' found in: {root_or_zip}")

    def _scan_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        if self.is_zip:
            with zipfile.ZipFile(self.root_or_zip) as zf:
                names = sorted(
                    name
                    for name in zf.namelist()
                    if not name.endswith("/") and name.endswith(self.suffix)
                )
            for name in names:
                label = self._parse_label(os.path.basename(name))
                samples.append((name, label))
        else:
            for current_root, _, files in os.walk(self.root_or_zip):
                for filename in sorted(files):
                    if filename.endswith(self.suffix):
                        path = os.path.join(current_root, filename)
                        label = self._parse_label(filename)
                        samples.append((path, label))
        return samples

    @staticmethod
    def _parse_label(filename: str) -> int:
        try:
            return int(filename.split("_")[0])
        except Exception as exc:
            raise ValueError(f"Unable to parse label from filename: {filename}") from exc

    def _get_zip_file(self) -> zipfile.ZipFile:
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.root_or_zip)
        return self._zip_file

    def __getitem__(self, index: int):
        sample_path, label = self.samples[index]
        if self.is_zip:
            with self._get_zip_file().open(sample_path) as fp:
                image = Image.open(io.BytesIO(fp.read())).convert("RGB")
        else:
            image = Image.open(sample_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        return len(self.samples)


def build_translated_loaders(
    translated_root: str,
    target: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    pretrained: bool,
    seed: int,
) -> Dict[str, DataLoader]:
    transform = build_transforms(image_size=image_size, pretrained=pretrained)
    translated_full = TranslatedDigitsDataset(translated_root, transform=transform)
    translated_train, translated_val = split_train_val(translated_full, val_ratio=val_ratio, seed=seed)
    tgt_test = get_digit_dataset(target, data_root, train=False, transform=transform)
    tgt_train = get_digit_dataset(target, data_root, train=True, transform=transform)

    return {
        "translated_train": build_loader(translated_train, batch_size, num_workers, shuffle=True),
        "translated_val": build_loader(translated_val, batch_size, num_workers, shuffle=False),
        "target_train": build_loader(tgt_train, batch_size, num_workers, shuffle=True),
        "target_test": build_loader(tgt_test, batch_size, num_workers, shuffle=False),
    }


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
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
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

    return running_loss / total, correct / total


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

    return running_loss / total, correct / total


def create_writer(output_dir: str):
    if SummaryWriter is None:
        return None
    log_dir = os.path.join(output_dir, "tensorboard")
    return SummaryWriter(log_dir=log_dir)


def log_scalars(writer, metrics: Dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
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
    for name, opt in (optimizers or {}).items():
        if name in checkpoint.get("optimizers", {}):
            opt.load_state_dict(checkpoint["optimizers"][name])
    for name, sch in (schedulers or {}).items():
        if name in checkpoint.get("schedulers", {}):
            sch.load_state_dict(checkpoint["schedulers"][name])
    return checkpoint


def save_json(path: str, payload: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.close()


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

