"""Assemble episode: concat clips + local TTS narration + burned subtitles."""
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from ...utils.system_check import get_ffmpeg_path

logger = logging.getLogger(__name__)


def _ffprobe(ffmpeg: str) -> str:
    p = Path(ffmpeg).with_name("ffprobe")
    return str(p) if p.exists() else (shutil.which("ffprobe") or "ffprobe")


def _dur(ffprobe: str, path: Path) -> float:
    try:
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return 0.0


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def _wrap(text: str, width: int = 18) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))[:200]


def _ordered(script):
    by_frame = {}
    for t in (getattr(script, "video_tasks", None) or []):
        if t.status != "completed" or not getattr(t, "video_url", None):
            continue
        if not getattr(t, "frame_id", None):
            continue
        prev = by_frame.get(t.frame_id)
        if prev is None or getattr(t, "model", "") == "motion_comic":
            by_frame[t.frame_id] = t
    return [(f, by_frame[f.id]) for f in (script.frames or []) if f.id in by_frame]


def assemble_episode(pipeline, script_id: str) -> str:
    script = pipeline.get_script(script_id)
    if not script:
        raise ValueError("Script not found")
    ordered = _ordered(script)
    if not ordered:
        raise ValueError("没有可用的视频片段，请先在「5. 动效生成」生成片段。")

    ffmpeg = get_ffmpeg_path()
    ffprobe = _ffprobe(ffmpeg)
    work = Path("output/export") / ("assemble_%s_%s" % (script_id[:8], uuid.uuid4().hex[:6]))
    work.mkdir(parents=True, exist_ok=True)

    try:
        from ...models.local import get_local_tts
        tts = get_local_tts()
    except Exception as exc:
        logger.warning("Local TTS unavailable: %s", exc)
        tts = None

    segs, cues, cursor = [], [], 0.0
    for idx, (frame, task) in enumerate(ordered):
        clip = Path("output") / str(task.video_url).lstrip("/")
        if not clip.is_file():
            continue
        clip_dur = float(getattr(task, "duration", 0) or 0) or _dur(ffprobe, clip) or 4.0
        text = (getattr(frame, "dialogue", "") or "").strip()
        wav = None
        audio_dur = 0.0
        if tts and text:
            wav = work / ("seg_%03d.wav" % idx)
            try:
                tts.synthesize(text, str(wav))
                audio_dur = _dur(ffprobe, wav)
            except Exception as exc:
                logger.warning("TTS failed for %s: %s", frame.id, exc)
                wav = None
        final_dur = max(clip_dur, audio_dur + 0.2) if audio_dur else clip_dur
        pad = max(0.0, final_dur - clip_dur)
        seg = work / ("seg_%03d.mp4" % idx)
        if wav and wav.is_file():
            fc = "[0:v]tpad=stop_mode=clone:stop_duration=%.2f,scale=1280:720,setsar=1[v];[1:a]apad[a]" % pad
            cmd = [ffmpeg, "-y", "-i", str(clip), "-i", str(wav), "-filter_complex", fc,
                   "-map", "[v]", "-map", "[a]", "-t", "%.2f" % final_dur,
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                   "-c:a", "aac", "-ar", "24000", "-ac", "1", str(seg)]
        else:
            cmd = [ffmpeg, "-y", "-i", str(clip), "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                   "-t", "%.2f" % final_dur, "-vf", "scale=1280:720,setsar=1",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                   "-c:a", "aac", "-ar", "24000", "-ac", "1", str(seg)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except Exception as exc:
            logger.warning("segment failed: %s", exc)
            continue
        if text:
            cues.append((cursor, cursor + final_dur, _wrap(text)))
        cursor += final_dur
        segs.append(seg)

    if not segs:
        raise RuntimeError("没有成功合成的片段")

    list_file = work / "list.txt"
    list_file.write_text("".join("file '%s'\n" % p.resolve() for p in segs), encoding="utf-8")
    concat = work / "concat.mp4"
    subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c", "copy", str(concat)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    srt = work / "subs.srt"
    srt.write_text("\n".join(
        "%d\n%s --> %s\n%s\n" % (i, _ts(a), _ts(b), t)
        for i, (a, b, t) in enumerate(cues, 1)
    ), encoding="utf-8")

    out_dir = Path("output/video")
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / ("episode_%s.mp4" % script_id[:8])
    if cues:
        style = "FontName=PingFang SC,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=1,Shadow=0"
        vf = "subtitles='%s':force_style='%s'" % (srt, style)
        try:
            subprocess.run([ffmpeg, "-y", "-i", str(concat), "-vf", vf,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "copy", str(final)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except Exception as exc:
            logger.warning("subtitle burn failed (%s); saving without subtitles", exc)
            shutil.copy(concat, final)
    else:
        shutil.copy(concat, final)

    rel = os.path.relpath(final, "output")
    script.merged_video_url = rel
    pipeline._save_data()
    return rel
