"""P1 video metadata and P2 frame sampling with blur scores."""
from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _ffprobe(args: List[str]) -> str:
    return subprocess.run(["ffprobe", "-v", "error", *args], check=True, capture_output=True,
                          text=True).stdout


def frame_times(path: str) -> np.ndarray:
    """Presentation time of every video frame, sorted. Read from packets, so no decode."""
    out = _ffprobe(["-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0", path])
    pts = np.array(sorted(float(x) for x in out.split() if x and x != "N/A"))
    return pts - pts[0] if len(pts) else pts


def probe_video(path: str, tolerance_s: float = 0.1, tolerance_effective_s: float = 0.15) -> Dict:
    """P1. Frame rate comes from the measured frame times, so variable frame rate is handled."""
    info = json.loads(_ffprobe(["-print_format", "json", "-show_streams", "-show_format", path]))
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    pts = frame_times(path)
    step = np.diff(pts)
    fps_measured = float(1 / np.median(step)) if len(step) else float("nan")
    return {
        "codec": v.get("codec_name"),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps_nominal": float(Fraction(v["r_frame_rate"])),
        "fps_measured": fps_measured,
        "is_vfr": bool(len(step) and np.std(step) / np.mean(step) > 0.02),
        "n_frames": int(len(pts)),
        "duration_s": float(info["format"]["duration"]),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
        "tolerance_frames": tolerance_s * fps_measured,
        "tolerance_frames_effective": tolerance_effective_s * fps_measured,
        "_pts": pts,
    }


def plan_frames(pts: np.ndarray, boundaries: np.ndarray, timeline_fps: float, window_s: float) -> pd.DataFrame:
    """Frames to keep: one per timeline tick, and every frame within window_s of a boundary."""
    duration = float(pts[-1]) if len(pts) else 0.0
    ticks = np.arange(0, duration + 1e-9, 1.0 / timeline_fps)
    tick_idx = np.unique(np.clip(np.searchsorted(pts, ticks), 0, len(pts) - 1))
    near = np.zeros(len(pts), dtype=bool)
    for b in boundaries:
        near |= np.abs(pts - b) <= window_s
    timeline = np.zeros(len(pts), dtype=bool)
    timeline[tick_idx] = True
    keep = np.flatnonzero(timeline | near)
    return pd.DataFrame({"frame_idx": keep, "t": pts[keep], "is_timeline": timeline[keep],
                         "near_boundary": near[keep]})


def extract_frames(video_path: str, plan: pd.DataFrame, out_dir: Path, width: int, quality: int) -> pd.DataFrame:
    """P2. Decode once, save the planned frames as JPEG, and score blur on each."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan.copy()
    plan["path"] = [str(out_dir / f"{i:06d}.jpg") for i in plan.frame_idx]
    blur_file = out_dir / "blur.json"
    if blur_file.exists() and all(Path(p).exists() for p in plan.path):
        blur = json.loads(blur_file.read_text())
        plan["blur"] = [blur.get(str(i), np.nan) for i in plan.frame_idx]
        return plan
    wanted = set(int(i) for i in plan.frame_idx)
    last = max(wanted) if wanted else -1
    cap = cv2.VideoCapture(str(video_path))
    blur: Dict[str, float] = {}
    idx = 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, quality])
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blur[str(idx)] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        idx += 1
    cap.release()
    blur_file.write_text(json.dumps(blur))
    plan["blur"] = [blur.get(str(i), np.nan) for i in plan.frame_idx]
    missing = plan.blur.isna().sum()
    if missing:
        plan = plan[plan.blur.notna()].reset_index(drop=True)
    return plan


def assign_segments(frames: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    """Segment that contains each frame time, [start, end). The last segment keeps its end."""
    seg = seg.sort_values("start_s")
    starts, ends, ids = seg.start_s.to_numpy(), seg.end_s.to_numpy(), seg.segment_id.to_numpy()
    k = np.searchsorted(starts, frames.t.to_numpy(), side="right") - 1
    inside = (k >= 0) & (frames.t.to_numpy() <= ends[np.clip(k, 0, len(ends) - 1)] + 1e-6)
    frames = frames.copy()
    frames["segment_id"] = np.where(inside, ids[np.clip(k, 0, len(ids) - 1)], "")
    return frames


def make_preview(video_path: str, out_path: Path, height: int = 360, crf: int = 30) -> Path:
    """Small H.264 copy for embedding in the notebook."""
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vf", f"scale=-2:{height}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", str(out_path)], check=True)
    return out_path
