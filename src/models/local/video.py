"""Local video generation.

Currently implemented: LTX-Video (text-to-video and first-frame image-to-video).
Wan2.1-VACE (reference-to-video / video editing) ships in Wan's native layout
and needs a one-time conversion to a diffusers pipeline; calling it raises an
explicit, actionable error instead of silently failing.
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from ..base import VideoGenModel
from .image import materialize_image, parse_size
from .paths import model_path
from .runtime import CACHE, device, dtype, low_vram, to_device

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 704
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 24


def _load_ltx_t2v():
    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained(
        str(model_path("ltx-video")), torch_dtype=dtype("bf16")
    )
    return to_device(pipe)


def _load_ltx_i2v():
    from diffusers import LTXImageToVideoPipeline

    pipe = LTXImageToVideoPipeline.from_pretrained(
        str(model_path("ltx-video")), torch_dtype=dtype("bf16")
    )
    return to_device(pipe)


class LocalVideoModel(VideoGenModel):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        logger.info(
            "LocalVideoModel ready (device=%s, low_vram=%s)", device(), low_vram()
        )

    def generate(
        self,
        prompt: str,
        output_path: str,
        img_path: str = None,
        img_url: str = None,
        model_name: str = None,
        duration: int = 5,
        **kwargs,
    ) -> Tuple[str, float]:
        name = (model_name or "").lower()
        if "vace" in name or kwargs.get("mode") in ("r2v", "v2v"):
            raise NotImplementedError(
                "本地 Wan2.1-VACE (R2V/V2V) 尚未接入：模型为 Wan 原生格式，需要先转换为 diffusers pipeline。"
                f" 已下载权重: {model_path('wan21-vace-1.3b', required=False)}。"
                " 在此之前请把 VIDEO_BACKEND 设为 dashscope 使用云端 R2V。"
            )

        start = time.time()
        _low = low_vram()
        _default_w = int(os.getenv("LOCAL_VIDEO_WIDTH") or (512 if _low else DEFAULT_WIDTH))
        _default_h = int(os.getenv("LOCAL_VIDEO_HEIGHT") or (320 if _low else DEFAULT_HEIGHT))
        width, height = parse_size(
            kwargs.get("size"), default=(_default_w, _default_h)
        )
        fps = int(kwargs.get("fps") or os.getenv("LOCAL_VIDEO_FPS") or (12 if _low else DEFAULT_FPS))
        max_frames = int(os.getenv("LOCAL_VIDEO_MAX_FRAMES") or (49 if _low else 121))
        num_frames = max(9, (int(duration * fps) // 8) * 8 + 1)
        num_frames = min(num_frames, max_frames)
        steps = int(os.getenv("LOCAL_VIDEO_STEPS") or (20 if _low else 30))
        seed = kwargs.get("seed")

        import torch

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed if seed is not None else int(time.time()) % (2**31))

        source = img_path or img_url
        local_image = materialize_image(source) if source else None
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        if local_image:
            from PIL import Image

            pipe = CACHE.get("ltx-i2v", _load_ltx_i2v)
            image = Image.open(local_image).convert("RGB").resize((width, height))
            result = pipe(
                image=image,
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                generator=generator,
            )
        else:
            pipe = CACHE.get("ltx-t2v", _load_ltx_t2v)
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=steps,
                generator=generator,
            )

        frames = result.frames[0]
        _export(frames, output_path, fps)
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        elapsed = time.time() - start
        logger.info("Local video generated in %.1fs -> %s", elapsed, output_path)
        return output_path, elapsed


def _export(frames, output_path: str, fps: int) -> None:
    tmp_path = output_path
    if output_path.lower().endswith(".mp4"):
        try:
            from diffusers.utils import export_to_video

            export_to_video(frames, tmp_path, fps=fps)
            return
        except Exception as exc:
            logger.warning("export_to_video failed (%s); falling back to imageio", exc)
    import imageio.v2 as imageio

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


_SINGLETON: Optional[LocalVideoModel] = None


def get_local_video_model(config: Dict[str, Any] = None) -> LocalVideoModel:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = LocalVideoModel(config)
    return _SINGLETON
