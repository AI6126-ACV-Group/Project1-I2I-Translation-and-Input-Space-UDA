import os
import zipfile
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from .common import (
    NUM_CLASSES,
    ResNet18FeatureNet,
    build_loader,
    ensure_dir,
    freeze_module,
    get_digit_dataset,
)


def build_gan_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def build_unpaired_digit_loaders(
    source: str,
    target: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    transform = build_gan_transforms(image_size=image_size)
    src_train = get_digit_dataset(source, data_root, train=True, transform=transform)
    tgt_train = get_digit_dataset(target, data_root, train=True, transform=transform)
    src_export = get_digit_dataset(source, data_root, train=True, transform=transform)

    return {
        "source_train": build_loader(src_train, batch_size, num_workers, shuffle=True),
        "target_train": build_loader(tgt_train, batch_size, num_workers, shuffle=True),
        "source_export": build_loader(src_export, batch_size, num_workers, shuffle=False),
    }


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7, bias=False),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]

        channels = base_channels
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        channels * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(channels * 2),
                    nn.ReLU(inplace=True),
                ]
            )
            channels *= 2

        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(channels))

        for _ in range(2):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        channels,
                        channels // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(channels // 2),
                    nn.ReLU(inplace=True),
                ]
            )
            channels //= 2

        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(channels, out_channels, kernel_size=7),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeastSquaresGanLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_value = 1.0 if target_is_real else 0.0
        target = torch.full_like(prediction, fill_value=target_value)
        return self.loss(prediction, target)


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    return images.mul(0.5).add(0.5).clamp(0.0, 1.0)


def low_frequency_fft_loss(real: torch.Tensor, reconstructed: torch.Tensor, ratio: float = 0.25) -> torch.Tensor:
    if ratio <= 0:
        return real.new_tensor(0.0)

    def extract_low_frequency_patch(images: torch.Tensor) -> torch.Tensor:
        gray = images.mean(dim=1, keepdim=True)
        fft = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"))
        magnitude = torch.log1p(torch.abs(fft))
        height, width = magnitude.shape[-2:]
        patch_h = max(1, int(round(height * ratio)))
        patch_w = max(1, int(round(width * ratio)))
        center_h = height // 2
        center_w = width // 2
        start_h = max(0, center_h - patch_h // 2)
        start_w = max(0, center_w - patch_w // 2)
        end_h = min(height, start_h + patch_h)
        end_w = min(width, start_w + patch_w)
        return magnitude[:, :, start_h:end_h, start_w:end_w]

    return F.l1_loss(extract_low_frequency_patch(real), extract_low_frequency_patch(reconstructed))


def load_frozen_classifier(checkpoint_path: str, device: torch.device) -> ResNet18FeatureNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ResNet18FeatureNet(num_classes=NUM_CLASSES, pretrained=False).to(device)
    if "modules" in checkpoint and "model" in checkpoint["modules"]:
        model.load_state_dict(checkpoint["modules"]["model"])
    else:
        model.load_state_dict(checkpoint)
    freeze_module(model)
    return model


def semantic_consistency_loss(
    teacher: nn.Module,
    fake_target: torch.Tensor,
    labels: torch.Tensor,
    image_size: int = 224,
) -> torch.Tensor:
    resized = F.interpolate(fake_target, size=(image_size, image_size), mode="bilinear", align_corners=False)
    logits = teacher(resized)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def save_translation_preview(
    output_dir: str,
    epoch: int,
    generator_a2b: nn.Module,
    generator_b2a: nn.Module,
    real_a: torch.Tensor,
    real_b: torch.Tensor,
    max_items: int = 8,
) -> str:
    ensure_dir(output_dir)
    real_a = real_a[:max_items]
    real_b = real_b[:max_items]
    fake_b = generator_a2b(real_a)
    fake_a = generator_b2a(real_b)
    cycle_a = generator_b2a(fake_b)
    cycle_b = generator_a2b(fake_a)

    rows = torch.cat(
        [
            denormalize_images(real_a.cpu()),
            denormalize_images(fake_b.cpu()),
            denormalize_images(cycle_a.cpu()),
            denormalize_images(real_b.cpu()),
            denormalize_images(fake_a.cpu()),
            denormalize_images(cycle_b.cpu()),
        ],
        dim=0,
    )
    grid = make_grid(rows, nrow=max_items)
    preview_path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
    save_image(grid, preview_path)
    return preview_path


@torch.no_grad()
def export_translated_digits(
    generator: nn.Module,
    loader: DataLoader,
    output_dir: str,
    device: torch.device,
    filename_suffix: str = "_fake_B.png",
    max_items: Optional[int] = None,
    zip_output: str = "",
) -> Dict[str, str]:
    ensure_dir(output_dir)
    generator.eval()

    exported = 0
    zip_file = None
    if zip_output:
        zip_parent = os.path.dirname(zip_output)
        if zip_parent:
            ensure_dir(zip_parent)
        zip_file = zipfile.ZipFile(zip_output, mode="w", compression=zipfile.ZIP_DEFLATED)

    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            fake_images = denormalize_images(generator(images)).cpu()
            for fake_image, label in zip(fake_images, labels.tolist()):
                if max_items is not None and exported >= max_items:
                    break
                filename = f"{label}_{exported:06d}{filename_suffix}"
                path = os.path.join(output_dir, filename)
                save_image(fake_image, path)
                if zip_file is not None:
                    zip_file.write(path, arcname=filename)
                exported += 1
            if max_items is not None and exported >= max_items:
                break
    finally:
        if zip_file is not None:
            zip_file.close()

    return {
        "output_dir": output_dir,
        "zip_output": zip_output,
        "num_exported": exported,
    }
