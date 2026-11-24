"""Local image generation (T2I / I2I / inpaint / Kontext edit) via diffusers.

Model routing:
    * text-to-image            -> FLUX.1-schnell (quality) or SDXL (low VRAM)
    * reference images         -> SDXL + IP-Adapter (subject/style consistency)
    * mask + image             -> SDXL inpainting
    * instruction edit         -> FLUX.1-Kontext-dev
"""

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .paths import model_path
from .runtime import (
    CACHE,
    device,
    dtype,
    env_bool,
    low_vram,
    release_memory,
    to_device,
    total_ram_gb,
)

logger = logging.getLogger(__name__)

FLUX_RAM_THRESHOLD_GB = 40.0


def _round_multiple(value: int, multiple: int = 64) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def parse_size(
    size: Optional[str], default: Tuple[int, int] = (1024, 576)
) -> Tuple[int, int]:
    if not size:
        return default
    text = str(size).replace("x", "*").replace("X", "*")
    try:
        width, height = (int(part) for part in text.split("*")[:2])
        return _round_multiple(width), _round_multiple(height)
    except Exception:
        return default


def fit_with_padding(image, width: int, height: int):
    from PIL import Image

    src = image.convert("RGB")
    ratio = min(width / src.width, height / src.height)
    new_w = max(1, int(round(src.width * ratio)))
    new_h = max(1, int(round(src.height * ratio)))
    resized = src.resize((new_w, new_h), Image.LANCZOS)
    try:
        fill = src.getpixel((0, 0))
    except Exception:
        fill = (255, 255, 255)
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


def materialize_image(raw: Optional[str]) -> Optional[str]:
    """Resolve a local path or download a remote image to a temp file."""
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("http://") or text.startswith("https://"):
        try:
            import tempfile

            import requests

            response = requests.get(text, timeout=60)
            response.raise_for_status()
            suffix = ".png"
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                if ext in text.lower():
                    suffix = ext
                    break
            fd, path = tempfile.mkstemp(suffix=suffix, prefix="lumenx_ref_")
            with os.fdopen(fd, "wb") as fh:
                fh.write(response.content)
            return path
        except Exception as exc:
            logger.warning("Failed to download reference image %s: %s", text, exc)
            return None
    candidates = [
        text,
        os.path.join("output", text.lstrip("/")),
        os.path.join("output", text),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    logger.warning("Reference image not found locally: %s", text)
    return None


def _generator(seed: Optional[int]):
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed if seed is not None else random.randint(0, 2**31 - 1))
    return gen


def _load_sdxl():
    from diffusers import StableDiffusionXLPipeline

    path = model_path("sdxl-base")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(path), torch_dtype=dtype("fp16"), use_safetensors=True
    )
    return to_device(pipe)


def _sdxl_base():
    return CACHE.get("sdxl-base", _load_sdxl)


def _sdxl_img2img():
    from diffusers import AutoPipelineForImage2Image

    return AutoPipelineForImage2Image.from_pipe(_sdxl_base())


def _load_sdxl_ip():
    pipe = _load_sdxl()
    # IP-Adapter installs its own attention processors and cannot run with the
    # SlicedAttnProcessor that low-VRAM mode enables.
    try:
        pipe.disable_attention_slicing()
    except Exception:
        pass
    adapter_dir = model_path("ip-adapter") / "sdxl_models"
    # sdxl_models/image_encoder is CLIP ViT-bigG (1664), which pairs with
    # ip-adapter_sdxl.safetensors. The *_vit-h / *_plus_* weights need the ViT-H
    # encoder in models/image_encoder (not downloaded by default).
    weight_name = os.getenv("LOCAL_IP_ADAPTER_WEIGHT") or "ip-adapter_sdxl.safetensors"
    pipe.load_ip_adapter(
        str(adapter_dir),
        subfolder="",
        weight_name=weight_name,
        image_encoder_folder="image_encoder",
    )
    pipe.set_ip_adapter_scale(float(os.getenv("LOCAL_IP_ADAPTER_SCALE", "0.7")))
    return pipe


def _load_sdxl_inpaint():
    from diffusers import StableDiffusionXLInpaintPipeline

    path = model_path("sdxl-inpaint")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        str(path), torch_dtype=dtype("fp16"), use_safetensors=True
    )
    return to_device(pipe)


def _load_flux():
    import torch
    from diffusers import FluxPipeline

    path = model_path("flux1-schnell")
    pipe = FluxPipeline.from_pretrained(str(path), torch_dtype=torch.bfloat16)
    return to_device(pipe)


def _load_kontext():
    import torch
    from diffusers import FluxKontextPipeline

    path = model_path("flux1-kontext-dev")
    pipe = FluxKontextPipeline.from_pretrained(str(path), torch_dtype=torch.bfloat16)
    return to_device(pipe)


def _pick_t2i() -> str:
    requested = (os.getenv("LOCAL_T2I_MODEL") or "auto").strip().lower()
    if requested in ("sdxl", "flux"):
        return requested
    ram = total_ram_gb()
    if not low_vram() and (ram is None or ram >= FLUX_RAM_THRESHOLD_GB):
        try:
            model_path("flux1-schnell").exists()
            return "flux"
        except FileNotFoundError:
            pass
    return "sdxl"


class LocalImageModel:
    """Drop-in ImageGenModel-compatible adapter backed by local diffusers pipelines."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        logger.info(
            "LocalImageModel ready (device=%s, low_vram=%s)", device(), low_vram()
        )

    def generate(
        self,
        prompt: str,
        output_path: str,
        ref_image_path: str = None,
        ref_image_paths: List[str] = None,
        model_name: str = None,
        size: str = None,
        batch_size: int = 1,
        negative_prompt: str = "",
        **kwargs,
    ) -> Tuple[str, float]:
        start = time.time()
        width, height = parse_size(size, default=(1024, 576))
        seed = kwargs.get("seed")
        generator = _generator(seed)

        refs: List[str] = []
        for raw in ([ref_image_path] if ref_image_path else []) + list(
            ref_image_paths or []
        ):
            resolved = materialize_image(raw)
            if resolved:
                refs.append(resolved)

        mask_path = materialize_image(
            kwargs.get("mask_path") or kwargs.get("mask_image_path")
        )
        name = (model_name or "").lower()
        mode = (kwargs.get("mode") or "").lower()
        wants_edit = "kontext" in name or mode == "edit"
        wants_inpaint = bool(mask_path) or "inpaint" in name

        strength = kwargs.get("strength")
        if strength is None:
            strength = kwargs.get("ref_strength")
        try:
            strength = float(strength) if strength is not None else 0.62
        except (TypeError, ValueError):
            strength = 0.62

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        try:
            if wants_inpaint and refs:
                image = self._run_inpaint(
                    prompt,
                    negative_prompt,
                    refs[0],
                    mask_path,
                    width,
                    height,
                    generator,
                )
            elif mode == "img2img" and refs:
                image = self._run_img2img(
                    prompt,
                    negative_prompt,
                    refs[0],
                    width,
                    height,
                    generator,
                    strength,
                )
            elif wants_edit and refs:
                image = self._run_kontext(prompt, refs[0], width, height, generator)
            elif refs:
                image = self._run_ip_adapter(
                    prompt, negative_prompt, refs, width, height, generator
                )
            else:
                image = self._run_t2i(prompt, negative_prompt, width, height, generator)

            image.save(output_path)
        finally:
            if low_vram() or env_bool("LOCAL_FREE_CACHE_AFTER", False):
                release_memory()

        elapsed = time.time() - start
        logger.info("Local image generated in %.1fs -> %s", elapsed, output_path)
        return output_path, elapsed

    # ---- individual paths -------------------------------------------------

    def _run_t2i(self, prompt, negative_prompt, width, height, generator):
        backend = _pick_t2i()
        steps = int(os.getenv("LOCAL_IMAGE_STEPS") or (4 if backend == "flux" else 26))
        if backend == "flux":
            pipe = CACHE.get("flux1-schnell", _load_flux)
            return pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=float(os.getenv("LOCAL_FLUX_GUIDANCE", "0.0")),
                generator=generator,
            ).images[0]
        pipe = CACHE.get("sdxl-base", _load_sdxl)
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=float(os.getenv("LOCAL_SDXL_GUIDANCE", "6.0")),
            generator=generator,
        ).images[0]

    def _run_img2img(
        self, prompt, negative_prompt, image_path, width, height, generator, strength
    ):
        from PIL import Image

        steps = int(os.getenv("LOCAL_IMG2IMG_STEPS") or 20)
        pipe = _sdxl_img2img()
        image = fit_with_padding(Image.open(image_path), width, height)
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=image,
            strength=max(0.2, min(0.9, strength)),
            num_inference_steps=steps,
            guidance_scale=float(os.getenv("LOCAL_SDXL_GUIDANCE", "6.0")),
            generator=generator,
        ).images[0]

    def _run_ip_adapter(self, prompt, negative_prompt, refs, width, height, generator):
        from PIL import Image

        steps = int(os.getenv("LOCAL_IMAGE_STEPS") or 26)
        pipe = CACHE.get("sdxl-ip-adapter", _load_sdxl_ip)
        ref_image = Image.open(refs[0]).convert("RGB")
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            ip_adapter_image=ref_image,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=float(os.getenv("LOCAL_SDXL_GUIDANCE", "6.0")),
            generator=generator,
        ).images[0]

    def _run_inpaint(
        self, prompt, negative_prompt, image_path, mask_path, width, height, generator
    ):
        from PIL import Image

        if not mask_path:
            raise ValueError(
                "本地 SDXL inpainting 需要 mask_path（mask_image_path）参数。"
            )
        steps = int(os.getenv("LOCAL_IMAGE_STEPS") or 30)
        pipe = CACHE.get("sdxl-inpaint", _load_sdxl_inpaint)
        image = Image.open(image_path).convert("RGB").resize((width, height))
        mask = Image.open(mask_path).convert("L").resize((width, height))
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=image,
            mask_image=mask,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=float(os.getenv("LOCAL_SDXL_GUIDANCE", "6.0")),
            generator=generator,
        ).images[0]

    def _run_kontext(self, prompt, image_path, width, height, generator):
        from PIL import Image

        ram = total_ram_gb()
        if ram is not None and ram < FLUX_RAM_THRESHOLD_GB:
            raise RuntimeError(
                f"FLUX.1-Kontext-dev 需要约 {FLUX_RAM_THRESHOLD_GB:.0f}GB+ 内存，当前仅 {ram:.0f}GB。"
                " 可改用 SDXL inpaint（mask_path）或把 IMAGE_BACKEND 切回 dashscope。"
            )
        steps = int(os.getenv("LOCAL_KONTEXT_STEPS") or 28)
        pipe = CACHE.get("flux1-kontext-dev", _load_kontext)
        image = Image.open(image_path).convert("RGB").resize((width, height))
        return pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=float(os.getenv("LOCAL_KONTEXT_GUIDANCE", "2.5")),
            generator=generator,
        ).images[0]


_SINGLETON: Optional[LocalImageModel] = None


def get_local_image_model(config: Dict[str, Any] = None) -> LocalImageModel:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = LocalImageModel(config)
    return _SINGLETON
