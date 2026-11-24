"""Local TTS via CosyVoice3 (subprocess worker, zero-shot / instruct cloning)."""

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

from .paths import model_path

logger = logging.getLogger(__name__)

DEFAULT_COSYVOICE_PYTHON = (
    "/Volumes/ExternalSSD/codespace/SuperJAI/projects/yue/.venv-cosyvoice/bin/python"
)
DEFAULT_COSYVOICE_CODE_DIR = (
    "/Volumes/ExternalSSD/codespace/SuperJAI/projects/yue/third_party/CosyVoice"
)


class LocalCosyVoiceTTS:
    """Drop-in replacement for ``src.audio.tts.TTSProcessor`` using local CosyVoice3.

    System voice ids (longxiaochun, ...) do not exist locally. A reference audio
    clip is required; pass a local wav path as ``voice`` or set
    ``LOCAL_TTS_REFERENCE_AUDIO``.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or str(model_path("Fun-CosyVoice3-0.5B-2512"))
        self.code_dir = os.getenv("COSYVOICE_CODE_DIR") or DEFAULT_COSYVOICE_CODE_DIR
        self.python = os.getenv("COSYVOICE_PYTHON") or DEFAULT_COSYVOICE_PYTHON
        self.reference_audio = os.getenv("LOCAL_TTS_REFERENCE_AUDIO") or ""
        self.reference_text = os.getenv("LOCAL_TTS_REFERENCE_TEXT") or ""
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        logger.info("LocalCosyVoiceTTS ready (model_dir=%s)", self.model_dir)

    # ---- worker lifecycle -------------------------------------------------

    def _ensure_worker(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        env = dict(os.environ)
        env["COSYVOICE_MODEL_DIR"] = self.model_dir
        env["COSYVOICE_CODE_DIR"] = self.code_dir
        worker = str(Path(__file__).with_name("cosyvoice_worker.py"))
        self._proc = subprocess.Popen(
            [self.python, worker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        return self._proc

    def _call(self, payload: dict) -> dict:
        with self._lock:
            proc = self._ensure_worker()
            proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                stderr = ""
                try:
                    stderr = proc.stderr.read()[-2000:]
                except Exception:
                    pass
                raise RuntimeError(
                    f"CosyVoice worker exited unexpectedly. stderr={stderr}"
                )
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError(f"CosyVoice worker error: {response.get('error')}")
            return response

    # ---- public API (matches TTSProcessor.synthesize) ---------------------

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        speech_rate: float = 1.0,
        pitch_rate: float = 1.0,
        volume: int = 50,
        instructions: Optional[str] = None,
        model_override: Optional[str] = None,
        family_override: Optional[str] = None,
    ) -> Tuple[str, float, str]:
        reference = self._resolve_reference(voice)
        started = time.time()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        want_mp3 = out.suffix.lower() == ".mp3"
        tmp_wav = Path(tempfile.mkdtemp(prefix="lumenx_tts_")) / "out.wav"
        response = self._call(
            {
                "text": text,
                "out_path": str(tmp_wav),
                "reference_audio": reference,
                "reference_text": self.reference_text,
                "instruct": instructions or "",
                "speed": max(0.5, min(2.0, float(speech_rate or 1.0))),
            }
        )

        if want_mp3:
            self._to_mp3(Path(response["out_path"]), out)
        else:
            os.replace(response["out_path"], out)

        elapsed_ms = (time.time() - started) * 1000
        return str(out), elapsed_ms, f"local-{int(started)}"

    def _resolve_reference(self, voice: Optional[str]) -> str:
        if voice and os.path.isfile(str(voice)):
            return str(voice)
        if self.reference_audio and os.path.isfile(self.reference_audio):
            return self.reference_audio
        raise ValueError(
            "本地 CosyVoice3 需要参考音频：请把 wav 路径作为 voice 传入，"
            "或设置 LOCAL_TTS_REFERENCE_AUDIO。"
        )

    @staticmethod
    def _to_mp3(src: Path, dst: Path) -> None:
        from ...utils.system_check import get_ffmpeg_path

        ffmpeg = get_ffmpeg_path()
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(src),
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(dst),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            src.unlink()
        except OSError:
            pass


_SINGLETON: Optional[LocalCosyVoiceTTS] = None


def get_local_tts(model_dir: Optional[str] = None) -> LocalCosyVoiceTTS:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = LocalCosyVoiceTTS(model_dir)
    return _SINGLETON
