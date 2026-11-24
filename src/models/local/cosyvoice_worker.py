#!/usr/bin/env python3
"""CosyVoice3 synthesis worker (JSON-lines over stdin/stdout).

Runs inside a dedicated CosyVoice virtualenv so the heavy inference stack never
leaks into the API process. One JSON request per line in, one JSON response per
line out.

Request: {"text", "out_path", "reference_audio", "reference_text", "instruct", "speed"}
Response: {"ok": true, "out_path", "duration_ms"} | {"ok": false, "error"}

Environment:
    COSYVOICE_MODEL_DIR  path to Fun-CosyVoice3-0.5B-2512
    COSYVOICE_CODE_DIR   path to the CosyVoice python package
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_PROTOCOL = sys.stdout


def _log(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}), file=_PROTOCOL, flush=True)


def main() -> int:
    sys.stdout = sys.stderr

    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != script_dir]

    model_dir = os.environ.get("COSYVOICE_MODEL_DIR", "")
    if not model_dir or not Path(model_dir).is_dir():
        _log("COSYVOICE_MODEL_DIR is not set or missing")
        return 2

    code_dir = os.environ.get("COSYVOICE_CODE_DIR", "")
    if code_dir:
        sys.path.insert(0, code_dir)
        matcha = Path(code_dir) / "third_party" / "Matcha-TTS"
        if matcha.is_dir():
            sys.path.insert(0, str(matcha))

    try:
        import torch
        import torchaudio
        from cosyvoice.cli.cosyvoice import CosyVoice3
    except Exception as exc:  # noqa: BLE001
        _log(f"cosyvoice import failed: {exc}")
        return 2

    try:
        model = CosyVoice3(model_dir)
    except Exception as exc:  # noqa: BLE001
        _log(f"cosyvoice init failed: {exc}")
        return 2

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            reference = request.get("reference_audio") or ""
            reference_text = request.get("reference_text") or ""
            instruct = request.get("instruct") or ""
            speed = float(request.get("speed") or 1.0)
            if not reference or not Path(reference).is_file():
                raise ValueError("reference_audio is required for local synthesis")

            if reference_text:
                generator = model.inference_zero_shot(
                    request["text"],
                    reference_text,
                    reference,
                    stream=False,
                    speed=speed,
                )
            elif instruct:
                generator = model.inference_instruct2(
                    request["text"], instruct, reference, stream=False, speed=speed
                )
            else:
                generator = model.inference_cross_lingual(
                    request["text"], reference, stream=False, speed=speed
                )

            chunks = []
            for output in generator:
                chunks.append(output["tts_speech"])
            if not chunks:
                raise ValueError("cosyvoice produced no audio")

            audio = torch.cat(chunks, dim=1)
            out_path = Path(request["out_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                str(out_path),
                audio,
                model.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )
            duration_ms = int(round(audio.shape[-1] * 1000 / model.sample_rate))
            print(
                json.dumps(
                    {"ok": True, "out_path": str(out_path), "duration_ms": duration_ms}
                ),
                file=_PROTOCOL,
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"{type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
