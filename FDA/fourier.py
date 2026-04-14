from typing import Literal

import torch
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


def prepare_image_for_fda(image, image_size: int, mode: Literal["L", "RGB"]) -> torch.Tensor:
    image = TF.resize(image, [image_size, image_size], interpolation=InterpolationMode.BICUBIC)
    image = image.convert(mode)
    return TF.to_tensor(image)


def apply_fda(source: torch.Tensor, target: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    if source.shape != target.shape:
        raise ValueError(f"Source/target tensor shape mismatch: {source.shape} vs {target.shape}")
    if beta <= 0:
        return source.clamp(0.0, 1.0)

    fft_src = torch.fft.fft2(source, dim=(-2, -1))
    fft_tgt = torch.fft.fft2(target, dim=(-2, -1))

    amp_src = torch.abs(fft_src)
    amp_tgt = torch.abs(fft_tgt)
    pha_src = torch.angle(fft_src)

    amp_src = torch.fft.fftshift(amp_src, dim=(-2, -1))
    amp_tgt = torch.fft.fftshift(amp_tgt, dim=(-2, -1))

    _, height, width = source.shape
    band = max(1, int(round(min(height, width) * beta)))
    center_h = height // 2
    center_w = width // 2
    h1 = max(0, center_h - band)
    h2 = min(height, center_h + band + 1)
    w1 = max(0, center_w - band)
    w2 = min(width, center_w + band + 1)

    amp_src[:, h1:h2, w1:w2] = amp_tgt[:, h1:h2, w1:w2]
    amp_src = torch.fft.ifftshift(amp_src, dim=(-2, -1))

    mixed_fft = amp_src * torch.exp(1j * pha_src)
    mixed = torch.fft.ifft2(mixed_fft, dim=(-2, -1)).real
    return mixed.clamp(0.0, 1.0)
