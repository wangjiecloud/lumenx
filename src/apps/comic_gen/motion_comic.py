"""Motion-comic generation: turn storyboard stills into short Ken-Burns clips.

This avoids the local video model entirely (fast + memory-light): each shot's
representative frame becomes a 4s clip with a slow zoom/pan, produced by FFmpeg.
"""

import logging
import os
import subprocess
import uuid
from pathlib import Path

from ...utils.system_check import get_ffmpeg_path
from .models import VideoTask

logger = logging.getLogger(__name__)


def _rendered_url(frame):
    unit = getattr(frame, "rendered_image_asset", None)
    variants = getattr(unit, "variants", None) or []
    if variants:
        if getattr(unit, "selected_id", None):
            chosen = next((v for v in variants if v.id == unit.selected_id), None)
            if chosen:
                return chosen.url
        return variants[-1].url
    return getattr(frame, "image_url", None) or getattr(
        frame, "rendered_image_url", None
    )


def _size_cat(shot_size: str) -> str:
    text = shot_size or ""
    if any(k in text for k in ("特写", "近景")):
        return "close"
    if any(k in text for k in ("全景", "远景")):
        return "wide"
    return "medium"


def _resolve_output_path(url: str) -> Path:
    output_root = Path("output").resolve()
    candidate = (output_root / str(url).lstrip("/")).resolve()
    if output_root not in candidate.parents and candidate != output_root:
        raise ValueError("image path escapes output dir")
    return candidate


def _effect(index: int, frames: int) -> str:
    if index % 3 == 0:
        return (
            "scale=1280:-2,zoompan=z='min(zoom+0.0015,1.18)':x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720:fps=24"
        )
    if index % 3 == 1:
        return (
            "scale=1280:-2,zoompan=z='if(eq(on,1),1.18,max(1.0,zoom-0.0015))':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720:fps=24"
        )
    return (
        "scale=1400:-2,zoompan=z='1.12':"
        f"x='(iw-iw/zoom)*on/{frames}':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720:fps=24"
    )


def create_motion_comic_clips(pipeline, script_id: str, clip_seconds: int = 4) -> int:
    script = pipeline.get_script(script_id)
    if not script:
        raise ValueError("Script not found")

    groups, current, prev_key = [], [], None
    for frame in script.frames or []:
        key = (frame.scene_id, _size_cat(frame.shot_size))
        if current and (key != prev_key or len(current) >= 3 or current[-1].dialogue):
            groups.append(current)
            current = []
        if _rendered_url(frame):
            current.append(frame)
        prev_key = key
    if current:
        groups.append(current)

    existing = {
        t.frame_id
        for t in (getattr(script, "video_tasks", None) or [])
        if getattr(t, "frame_id", None) and getattr(t, "model", "") == "motion_comic"
    }

    ffmpeg = get_ffmpeg_path()
    out_dir = Path("output/video")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = clip_seconds * 24
    created = 0

    for index, group in enumerate(groups):
        target = next((f for f in group if _rendered_url(f)), None)
        if not target or target.id in existing:
            continue
        url = _rendered_url(target)
        try:
            src = _resolve_output_path(url)
        except ValueError:
            continue
        if not src.is_file():
            continue

        task_id = str(uuid.uuid4())
        out_path = out_dir / f"mc_{target.id}_{task_id[:8]}.mp4"
        vf = _effect(index, frames)
        cmd = [
            ffmpeg,
            "-y",
            "-loop",
            "1",
            "-i",
            str(src),
            "-t",
            str(clip_seconds),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "24",
            str(out_path),
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception as exc:
            logger.warning("Motion comic ffmpeg failed for %s: %s", target.id, exc)
            continue

        prompt = (
            target.image_prompt_en
            or target.image_prompt
            or target.visual_description
            or target.action_description
            or ""
        )
        if not getattr(script, "video_tasks", None):
            script.video_tasks = []
        script.video_tasks.append(
            VideoTask(
                id=task_id,
                project_id=script_id,
                frame_id=target.id,
                image_url=url,
                prompt=prompt,
                status="completed",
                video_url=os.path.relpath(out_path, "output"),
                duration=clip_seconds,
                model="motion_comic",
            )
        )
        created += 1

    if created:
        pipeline._save_data()
        logger.info("Motion comic: created %d clips", created)
    return created
