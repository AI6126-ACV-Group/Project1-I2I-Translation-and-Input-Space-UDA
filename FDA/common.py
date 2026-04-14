import json
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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


class ResNet18FeatureNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
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


def train_one_epoch(
    model: nn.Module,
    loader,
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
    loader,
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
    map_location="cpu",
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


def save_progress(output_dir: str, history: List[Dict], summary: Dict) -> None:
    save_json(
        os.path.join(output_dir, "progress.json"),
        {
            "history": history,
            "summary": summary,
        },
    )
