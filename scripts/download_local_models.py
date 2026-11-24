#!/usr/bin/env python3
"""Download local model weights for LumenX local-generation capability.

Usage (run on a machine with internet access to huggingface.co):

    pip install -U "huggingface_hub>=0.23"
    python download_local_models.py --list
    python download_local_models.py --tier standard --dest ./lumenx_local_models

Then copy the staging dir into the model root on the LumenX machine:

    rsync -av --partial ./lumenx_local_models/ \
        /Volumes/ExternalSSD/codespace/SuperJAI/projects/yue/data/models/

Layout produced (model root / <category> / <key>):

    data/models/
      t2i/   sdxl-base, flux1-schnell, qwen-image
      i2i/   sdxl-inpaint, flux1-kontext-dev, qwen-image-edit, ip-adapter, ip-adapter-faceid
      video/ ltx-video, wan21-t2v-1.3b, wan21-i2v-14b-720p, wan21-vace-1.3b, ...
      asr/   whisper-large-v3-turbo, faster-whisper-large-v3
      post/  realesrgan, birefnet, sam2-hiera-large

Tiers:
    core      Minimum end-to-end pipeline, runs on Apple Silicon / modest GPUs.
    standard  Full LumenX feature set incl. character-consistency + cleanup. (default)
    all       Adds heavy optional models (Wan2.2 MoE, HunyuanVideo, VACE-14B, Qwen-Image).

Existing models (already in the model root, NOT downloaded here):
    Fun-CosyVoice3-0.5B-2512        (TTS + voice clone)
    Qwen2.5-VL-3B-Instruct-4bit     (VLM)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_DEST = "./lumenx_local_models"
MODEL_ROOT_HINT = "/Volumes/ExternalSSD/codespace/SuperJAI/projects/yue/data/models"
COMMON_IGNORE = ["*.md", ".gitattributes", "*.gif"]


@dataclass
class ModelSpec:
    key: str
    category: str
    repo: str
    capability: str
    tier: str
    approx_gb: float
    note: str = ""
    ignore: list = field(default_factory=list)
    url: str = ""
    archive: bool = False


REGISTRY = [
    ModelSpec(
        "sdxl-base",
        "t2i",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "text-to-image",
        "core",
        14.0,
        "base T2I, light",
        ignore=["sd_xl_base_1.0.safetensors"],
    ),
    ModelSpec(
        "flux1-schnell",
        "t2i",
        "black-forest-labs/FLUX.1-schnell",
        "text-to-image",
        "standard",
        34.0,
        "high quality T2I, Apache-2.0, 4-step",
        ignore=["flux1-schnell.safetensors", "flux1-schnell-fp8.safetensors"],
    ),
    ModelSpec(
        "qwen-image",
        "t2i",
        "Qwen/Qwen-Image",
        "text-to-image",
        "all",
        40.0,
        "top quality T2I, heavy",
    ),
    ModelSpec(
        "ip-adapter",
        "i2i",
        "h94/IP-Adapter",
        "reference-consistency",
        "standard",
        4.5,
        "SDXL IP-Adapter + ViT-H image encoder (excludes SD1.5 models/)",
        ignore=["models/*"],
    ),
    ModelSpec(
        "ip-adapter-faceid",
        "i2i",
        "h94/IP-Adapter-FaceID",
        "face-consistency",
        "standard",
        1.5,
        "character face consistency, SDXL only (excludes SD1.5)",
        ignore=["*sd15*"],
    ),
    ModelSpec(
        "antelopev2",
        "i2i",
        "",
        "face-embedding",
        "standard",
        0.35,
        "InsightFace antelopev2 (detect + embedding), required by IP-Adapter-FaceID",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
        archive=True,
    ),
    ModelSpec(
        "sdxl-inpaint",
        "i2i",
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "image-edit",
        "standard",
        7.0,
        "inpainting for frame retouch",
    ),
    ModelSpec(
        "flux1-kontext-dev",
        "i2i",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "image-edit",
        "standard",
        34.0,
        "instruction-based image edit, non-commercial",
        ignore=["flux1-kontext-dev.safetensors"],
    ),
    ModelSpec(
        "qwen-image-edit",
        "i2i",
        "Qwen/Qwen-Image-Edit",
        "image-edit",
        "all",
        40.0,
        "instruction-based edit, heavy",
    ),
    ModelSpec(
        "ltx-video",
        "video",
        "Lightricks/LTX-Video",
        "t2v+i2v",
        "core",
        15.0,
        "fast DiT video, best for Apple Silicon",
        ignore=["ltx-video-*.safetensors", "cctxt-*.safetensors"],
    ),
    ModelSpec(
        "wan21-t2v-1.3b",
        "video",
        "Wan-AI/Wan2.1-T2V-1.3B",
        "t2v",
        "core",
        18.0,
        "light Wan T2V",
    ),
    ModelSpec(
        "wan21-i2v-14b-720p",
        "video",
        "Wan-AI/Wan2.1-I2V-14B-720P",
        "i2v",
        "standard",
        43.0,
        "quality I2V, needs strong GPU",
    ),
    ModelSpec(
        "wan21-vace-1.3b",
        "video",
        "Wan-AI/Wan2.1-VACE-1.3B",
        "r2v+v2v",
        "standard",
        18.0,
        "reference-to-video / video edit, light",
    ),
    ModelSpec(
        "wan21-vace-14b",
        "video",
        "Wan-AI/Wan2.1-VACE-14B",
        "r2v+v2v",
        "all",
        33.0,
        "VACE heavy",
    ),
    ModelSpec(
        "wan22-t2v-a14b",
        "video",
        "Wan-AI/Wan2.2-T2V-A14B",
        "t2v",
        "all",
        40.0,
        "Wan2.2 MoE T2V",
    ),
    ModelSpec(
        "wan22-i2v-a14b",
        "video",
        "Wan-AI/Wan2.2-I2V-A14B",
        "i2v",
        "all",
        40.0,
        "Wan2.2 MoE I2V",
    ),
    ModelSpec(
        "hunyuanvideo",
        "video",
        "tencent/HunyuanVideo",
        "t2v",
        "all",
        25.0,
        "Tencent T2V, heavy",
    ),
    ModelSpec(
        "whisper-large-v3-turbo",
        "asr",
        "openai/whisper-large-v3-turbo",
        "speech-to-text",
        "standard",
        1.6,
        "transcribe reference audio for better clone",
    ),
    ModelSpec(
        "faster-whisper-large-v3",
        "asr",
        "Systran/faster-whisper-large-v3",
        "speech-to-text",
        "all",
        3.0,
        "CTranslate2 format (NOT PyTorch), faster inference",
    ),
    ModelSpec(
        "realesrgan",
        "post",
        "ai-forever/Real-ESRGAN",
        "upscale",
        "standard",
        0.1,
        "image/video frame super-resolution",
    ),
    ModelSpec(
        "birefnet",
        "post",
        "ZhengPeng7/BiRefNet",
        "matting",
        "standard",
        0.9,
        "background removal / matting",
    ),
    ModelSpec(
        "sam2-hiera-large",
        "post",
        "facebook/sam2-hiera-large",
        "segmentation",
        "all",
        0.9,
        "promptable segmentation",
    ),
]

TIER_ORDER = {"core": 0, "standard": 1, "all": 2}


def human_gb(n: float) -> str:
    return f"{n:.1f} GB"


def select_specs(args) -> list:
    specs = REGISTRY
    if args.category:
        cats = {c.strip() for c in args.category.split(",") if c.strip()}
        specs = [s for s in specs if s.category in cats]
    tier = TIER_ORDER[args.tier]
    specs = [s for s in specs if TIER_ORDER[s.tier] <= tier]
    if args.only:
        keys = {k.strip() for k in args.only.split(",") if k.strip()}
        unknown = keys - {s.key for s in REGISTRY}
        if unknown:
            sys.exit(f"Unknown model key(s): {', '.join(sorted(unknown))}. Use --list.")
        specs = [s for s in REGISTRY if s.key in keys]
    if args.exclude:
        drop = {k.strip() for k in args.exclude.split(",") if k.strip()}
        specs = [s for s in specs if s.key not in drop]
    return specs


def cmd_list(args) -> None:
    specs = select_specs(args)
    total = 0.0
    print(f"{'key':<24} {'tier':<10} {'category':<7} {'~size':>8}  repo")
    print("-" * 110)
    for s in specs:
        total += s.approx_gb
        print(
            f"{s.key:<24} {s.tier:<10} {s.category:<7} {human_gb(s.approx_gb):>8}  {s.repo or s.url}"
        )
        if s.note:
            print(f"{'':>24} └─ {s.note} [{s.capability}]")
    print("-" * 110)
    print(f"{len(specs)} models, approx total {human_gb(total)}")


def download_one(spec: ModelSpec, dest_root: Path, token, endpoint: str) -> dict:
    start = time.time()

    if spec.url:
        import shutil
        import urllib.request
        import zipfile
        from urllib.parse import urlparse

        cat_dir = dest_root / spec.category
        cat_dir.mkdir(parents=True, exist_ok=True)
        archive_path = cat_dir / Path(urlparse(spec.url).path).name
        with (
            urllib.request.urlopen(spec.url, timeout=300) as resp,
            open(archive_path, "wb") as fh,
        ):
            shutil.copyfileobj(resp, fh)
        if spec.archive:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(cat_dir)
            archive_path.unlink()
        target = cat_dir / spec.key
        size = (
            sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            if target.exists()
            else 0
        )
        return {
            "key": spec.key,
            "category": spec.category,
            "repo": spec.url,
            "local_dir": str(target),
            "bytes": size,
            "seconds": round(time.time() - start, 1),
            "path": str(target),
        }

    from huggingface_hub import snapshot_download

    target = dest_root / spec.category / spec.key
    target.mkdir(parents=True, exist_ok=True)
    ignore = sorted(set(COMMON_IGNORE + spec.ignore))
    path = snapshot_download(
        repo_id=spec.repo,
        local_dir=str(target),
        ignore_patterns=ignore,
        token=token,
        endpoint=endpoint,
        max_workers=4,
    )
    elapsed = time.time() - start
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    return {
        "key": spec.key,
        "category": spec.category,
        "repo": spec.repo,
        "local_dir": str(target),
        "bytes": size,
        "seconds": round(elapsed, 1),
        "path": str(path),
    }


def cmd_download(args) -> None:
    token = (
        args.token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    endpoint = args.endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT
    dest_root = Path(args.dest).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    specs = select_specs(args)

    print(f"endpoint : {endpoint}")
    print(f"dest     : {dest_root}")
    print(f"models   : {len(specs)}  (tier={args.tier})")
    if args.dry_run:
        for s in specs:
            print(
                f"  [dry-run] {s.category}/{s.key}  <-  {s.repo}  ({human_gb(s.approx_gb)})"
            )
        return

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit(
            "huggingface_hub not installed. Run: pip install -U 'huggingface_hub>=0.23'"
        )

    manifest_path = dest_root / "_manifest.json"
    manifest = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    done_keys = {m.get("key") for m in manifest if m.get("bytes")}

    failures = []
    for i, spec in enumerate(specs, 1):
        if spec.key in done_keys and not args.force:
            print(
                f"[{i}/{len(specs)}] {spec.key}: already in manifest, skip (use --force to redo)"
            )
            continue
        print(
            f"[{i}/{len(specs)}] {spec.key} <- {spec.repo}  (approx {human_gb(spec.approx_gb)})"
        )
        try:
            info = download_one(spec, dest_root, token, endpoint)
            manifest = [m for m in manifest if m.get("key") != spec.key]
            manifest.append(info)
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
            print(
                f"    ok: {info['bytes'] / 1e9:.2f} GB in {info['seconds']}s -> {info['local_dir']}"
            )
        except KeyboardInterrupt:
            print("\nInterrupted. Partial files stay in place; rerun to resume.")
            raise
        except Exception as exc:
            failures.append((spec.key, str(exc)))
            print(f"    FAILED: {exc}")

    print()
    print("=" * 70)
    total_bytes = sum(m.get("bytes", 0) for m in manifest)
    print(f"downloaded/verified: {len(manifest)} models, {total_bytes / 1e9:.2f} GB")
    if failures:
        print("FAILED:")
        for key, err in failures:
            print(f"  - {key}: {err}")
        print("Rerun the same command to resume; failures are retried.")
    print()
    print("Next: copy the staging dir into the model root on the LumenX machine, e.g.")
    print(f"  rsync -av --partial {dest_root}/ {MODEL_ROOT_HINT}/")
    print("  (layout: <category>/<key>/ merges into the model root as subfolders)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download local model weights for LumenX"
    )
    parser.add_argument(
        "--tier", choices=["core", "standard", "all"], default="standard"
    )
    parser.add_argument("--category", help="filter by category: t2i,i2i,video,asr,post")
    parser.add_argument("--only", help="comma-separated model keys (see --list)")
    parser.add_argument("--exclude", help="comma-separated model keys to skip")
    parser.add_argument("--dest", default=DEFAULT_DEST)
    parser.add_argument("--endpoint", help=f"HF endpoint (default {DEFAULT_ENDPOINT})")
    parser.add_argument("--token", help="HF token (or env HF_TOKEN)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="redownload even if in manifest"
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        cmd_list(args)
    else:
        cmd_download(args)


if __name__ == "__main__":
    main()
