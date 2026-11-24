"""Local post-processing helpers: super-resolution (Real-ESRGAN) and matting (BiRefNet)."""

import logging
import os
from typing import Optional

from .paths import model_path
from .runtime import CACHE, device, dtype, to_device

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Real-ESRGAN (RRDBNet architecture, matches RealESRGAN_x4.pth)
# --------------------------------------------------------------------------
def _build_rrdb_arch():
    import torch
    from torch import nn

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(
            self,
            num_in_ch=3,
            num_out_ch=3,
            scale=4,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
        ):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(
                *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
            )
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            import torch.nn.functional as F

            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(
                self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            feat = self.lrelu(
                self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
            )
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet


def _load_realesrgan():
    import torch

    RRDBNet = _build_rrdb_arch()
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32
    )
    weight = model_path("realesrgan") / "RealESRGAN_x4.pth"
    state = torch.load(str(weight), map_location="cpu")
    if isinstance(state, dict) and "params_ema" in state:
        state = state["params_ema"]
    model.load_state_dict(state, strict=True)
    return to_device(model.eval())


def upscale_image(input_path: str, output_path: str) -> str:
    """4x upscale a single image with Real-ESRGAN."""
    import numpy as np
    import torch
    from PIL import Image

    model = CACHE.get("realesrgan", _load_realesrgan)
    image = Image.open(input_path).convert("RGB")
    tensor = (
        torch.from_numpy(np.array(image)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    )
    tensor = tensor.to(device())
    with torch.no_grad():
        out = model(tensor).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((out * 255.0).round().astype("uint8")).save(output_path)
    return output_path


# --------------------------------------------------------------------------
# BiRefNet matting (transformers remote code, bundled in the model dir)
# --------------------------------------------------------------------------
def _ensure_kornia_shim() -> None:
    """BiRefNet's remote code imports ``kornia.filters.laplacian`` at module
    level, but only uses it inside its training loss. Avoid the heavy
    kornia/kornia-rs dependency (Rust build) by providing a no-op shim when the
    real package is unavailable."""
    try:
        import kornia  # noqa: F401

        return
    except Exception:
        pass
    import sys
    import types

    import torch

    kornia_mod = types.ModuleType("kornia")
    filters_mod = types.ModuleType("kornia.filters")

    def laplacian(image, kernel_size=3):
        return torch.zeros_like(image)

    filters_mod.laplacian = laplacian
    kornia_mod.filters = filters_mod
    sys.modules.setdefault("kornia", kornia_mod)
    sys.modules.setdefault("kornia.filters", filters_mod)


def _load_birefnet():
    _ensure_kornia_shim()
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        str(model_path("birefnet")), trust_remote_code=True
    )
    return to_device(model.eval())


def remove_background(input_path: str, output_path: str, size: int = 1024) -> str:
    """Produce an RGBA PNG with the background made transparent."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    model = CACHE.get("birefnet", _load_birefnet)
    image = Image.open(input_path).convert("RGB")
    original_size = image.size
    preprocess = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = (
        preprocess(image)
        .unsqueeze(0)
        .to(device=device(), dtype=next(model.parameters()).dtype)
    )
    with torch.no_grad():
        preds = model(tensor)[-1].sigmoid().cpu()
    mask = transforms.ToPILImage()(preds[0].squeeze()).resize(original_size)
    result = image.convert("RGBA")
    result.putalpha(mask)
    result.save(output_path)
    return output_path
