#!/usr/bin/env python3
"""Local model readiness checker for LumenX.

Levels:
    paths    verify every model directory and required files exist (no torch)
    runtime  import torch/diffusers and report device / dtype / memory / backend
    smoke    run a minimal inference per selected model

Examples:
    python scripts/check_local_models.py --level paths
    python scripts/check_local_models.py --level runtime
    python scripts/check_local_models.py --level smoke --only sdxl-base,realesrgan
"""

import argparse
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cmd_paths(args):
    from src.models.local.paths import check_models, models_dir

    print(f"models dir: {models_dir()}\n")
    keys = [k.strip() for k in args.only.split(",")] if args.only else None
    results = check_models(keys)
    width = max(len(k) for k in results)
    ok_count = 0
    for key, (ok, missing) in results.items():
        ok_count += int(ok)
        status = "OK" if ok else f"MISSING {missing}"
        print(f"  {key:<{width}}  {status}")
    print(f"\n{ok_count}/{len(results)} models ready")
    return 0 if ok_count == len(results) else 1


def cmd_runtime(args):
    from src.models.local.runtime import describe, local_enabled

    info = describe()
    print("runtime:")
    for key, value in info.items():
        print(f"  {key:<18} {value}")
    print("\nbackend routing:")
    for kind in ("image", "video", "tts"):
        print(f"  {kind:<6} -> {'local' if local_enabled(kind) else 'dashscope'}")
    return 0 if info["deps"] else 1


def _smoke_image(pipe_fn, **kwargs) -> str:
    import importlib

    return pipe_fn(**kwargs)


def cmd_smoke(args):
    workdir = Path(tempfile.mkdtemp(prefix="lumenx_smoke_"))
    only = [k.strip() for k in args.only.split(",")] if args.only else None

    tests = {
        "sdxl-base": _smoke_sdxl,
        "sdxl-inpaint": _smoke_inpaint,
        "ip-adapter": _smoke_ip_adapter,
        "realesrgan": _smoke_realesrgan,
        "birefnet": _smoke_birefnet,
        "whisper-large-v3-turbo": _smoke_whisper,
        "Fun-CosyVoice3-0.5B-2512": _smoke_tts,
    }
    keys = only or list(tests.keys())
    failures = 0
    for key in keys:
        fn = tests.get(key)
        if not fn:
            print(f"[SKIP] {key}: no smoke test (heavy / deferred)")
            continue
        start = time.time()
        try:
            detail = fn(workdir)
            print(f"[PASS] {key} ({time.time() - start:.1f}s) {detail}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(
                f"[FAIL] {key} ({time.time() - start:.1f}s) {type(exc).__name__}: {exc}"
            )
            if args.verbose:
                traceback.print_exc()
    print(f"\nworkdir: {workdir}")
    return 0 if failures == 0 else 1


def _smoke_sdxl(workdir):
    from PIL import Image

    from src.models.local import get_local_image_model

    model = get_local_image_model({})
    out = workdir / "sdxl.png"
    model.generate("a red apple on a table, simple", str(out), size="512*512", seed=1)
    assert Image.open(out).size == (512, 512)
    return f"-> {out}"


def _smoke_inpaint(workdir):
    from PIL import Image, ImageDraw

    from src.models.local import get_local_image_model

    base = workdir / "inpaint_src.png"
    mask = workdir / "inpaint_mask.png"
    image = Image.new("RGB", (512, 512), (40, 120, 200))
    image.save(base)
    m = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(m).ellipse((176, 176, 336, 336), fill=255)
    m.save(mask)
    out = workdir / "inpaint_out.png"
    get_local_image_model({}).generate(
        "a yellow sun",
        str(out),
        ref_image_path=str(base),
        mask_path=str(mask),
        size="512*512",
        seed=2,
        model_name="sdxl-inpaint",
    )
    assert Image.open(out).size == (512, 512)
    return f"-> {out}"


def _smoke_ip_adapter(workdir):
    from PIL import Image

    from src.models.local import get_local_image_model

    ref = workdir / "ip_ref.png"
    Image.new("RGB", (512, 512), (200, 60, 60)).save(ref)
    out = workdir / "ip_out.png"
    get_local_image_model({}).generate(
        "a character in the same color palette",
        str(out),
        ref_image_paths=[str(ref)],
        size="512*512",
        seed=3,
    )
    assert Image.open(out).size == (512, 512)
    return f"-> {out}"


def _smoke_realesrgan(workdir):
    from PIL import Image

    from src.models.local import upscale_image

    src = workdir / "tiny.png"
    Image.new("RGB", (64, 64), (10, 200, 90)).save(src)
    out = workdir / "upscaled.png"
    upscale_image(str(src), str(out))
    assert Image.open(out).size == (256, 256), Image.open(out).size
    return f"-> {out}"


def _smoke_birefnet(workdir):
    from PIL import Image

    from src.models.local import remove_background

    src = workdir / "matte_src.png"
    Image.new("RGB", (256, 256), (250, 250, 250)).save(src)
    out = workdir / "matte.png"
    remove_background(str(src), str(out))
    assert Image.open(out).mode == "RGBA"
    return f"-> {out}"


def _smoke_whisper(workdir):
    import wave
    import struct

    from src.models.local import transcribe

    wav = workdir / "tone.wav"
    with wave.open(str(wav), "w") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(16000)
        fh.writeframes(b"".join(struct.pack("<h", 0) for _ in range(16000)))
    text = transcribe(str(wav))
    return f"transcribed {len(text)} chars"


def _smoke_tts(workdir):
    from src.models.local import get_local_tts

    reference = os.getenv("LOCAL_TTS_REFERENCE_AUDIO") or (
        "/Volumes/ExternalSSD/codespace/SuperJAI/projects/yue/data/assets/ziyue/voice_reference.wav"
    )
    if not os.path.isfile(reference):
        raise FileNotFoundError(
            f"需要参考音频，设置 LOCAL_TTS_REFERENCE_AUDIO（当前 {reference} 不存在）"
        )
    out = workdir / "tts.wav"
    tts = get_local_tts()
    path, elapsed_ms, req = tts.synthesize(
        "你好，这是本地语音合成的测试。", str(out), voice=reference
    )
    return f"-> {path} ({elapsed_ms:.0f}ms, {req})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", choices=["paths", "runtime", "smoke"], default="paths"
    )
    parser.add_argument("--only", help="comma-separated model keys")
    parser.add_argument("--models-dir", help="override LUMENX_MODELS_DIR")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.models_dir:
        os.environ["LUMENX_MODELS_DIR"] = args.models_dir

    if args.level == "paths":
        return cmd_paths(args)
    if args.level == "runtime":
        return cmd_runtime(args)
    return cmd_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
