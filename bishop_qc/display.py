"""Notebook views: clickable videos, segment cards, timelines, overlays and cost tables.

Colors follow a validated categorical palette (first three slots are safe for color blind
readers in any pair). Hand states use one blue ramp, light to dark, with gray for unknown.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .video import make_preview

SERIES = {"hand_motion": "#2a78d6", "camera_turn": "#eb6834", "forward": "#1baf7a"}
STATE_COLORS = {"contact": "#1c5cab", "no_contact": "#9ec5f4", "detected": "#d9d8d3", "unknown": "#f3f2ef"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SEG_FILL = ("#cde2fb", "#e8f1fd")
LITERAL_FILL = "#e6e5e1"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)


# Section 6: inputs -----------------------------------------------------------

def video_gallery(recordings: pd.DataFrame, segments: pd.DataFrame, cache_dir: Path,
                  height: int = 270, crf: int = 32):
    """One player per recording. Click a segment row to jump the player to that segment."""
    from IPython.display import HTML

    blocks = []
    for r in recordings.itertuples():
        prev = make_preview(r.path, Path(cache_dir) / "previews" / f"{r.recording_id}.mp4", height, crf)
        b64 = base64.b64encode(Path(prev).read_bytes()).decode()
        vid = f"vid_{r.recording_id}"
        rows = []
        for s in segments[segments.recording_id == r.recording_id].itertuples():
            rows.append(
                f"<tr style='cursor:pointer' onclick=\"var v=document.getElementById('{vid}');"
                f"v.currentTime={s.start_s};v.play();setTimeout(function(){{v.pause()}},"
                f"{int((s.end_s - s.start_s) * 1000)});\">"
                f"<td>{s.segment_index}</td><td>{s.start_s:.1f}</td><td>{s.end_s:.1f}</td>"
                f"<td>{html.escape(s.caption)}</td></tr>")
        blocks.append(f"""
<div style="display:flex;gap:16px;align-items:flex-start;margin:0 0 24px 0;font-family:sans-serif;font-size:12px;color:{INK}">
  <div style="flex:0 0 auto">
    <div style="font-weight:600;margin-bottom:4px">{html.escape(r.recording_id)}</div>
    <video id="{vid}" controls preload="metadata" style="height:{height}px;max-width:100%;background:#000">
      <source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>
  </div>
  <div style="flex:1 1 360px;max-height:{height + 24}px;overflow:auto;border:1px solid {GRID}">
    <table style="border-collapse:collapse;width:100%">
      <thead style="position:sticky;top:0;background:{SURFACE};color:{INK2}">
        <tr><th>#</th><th>start s</th><th>end s</th><th style="text-align:left">caption (click to play)</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table>
  </div>
</div>""")
    return HTML("".join(blocks))


def grab_frames(video_path: str, times: Sequence[float], width: int = 480) -> List[np.ndarray]:
    """RGB frames at the given times, by seeking."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, frame = cap.read()
        if not ok:
            out.append(np.zeros((int(width * 9 / 16), width, 3), np.uint8))
            continue
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def segment_card(seg_row: pd.Series, rec_row: pd.Series, n_frames: int = 4, width: int = 320):
    """Everything known about one segment, with frames from inside it."""
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display

    fields = ["segment_id", "start_s", "end_s", "caption", "annotator_id", "accepted_type",
              "caption_created_at", "caption_updated_at"]
    rows = "".join(f"<tr><td style='color:{INK2};padding-right:12px'>{f}</td><td>{html.escape(str(seg_row[f]))}</td></tr>"
                   for f in fields if f in seg_row)
    sugg = json.loads(seg_row["suggestions"]) if isinstance(seg_row.get("suggestions"), str) else []
    sugg_html = "".join(f"<li>{html.escape(s)}</li>" for s in sugg)
    synth = seg_row.get("synthetic_fields", "")
    display(HTML(f"""
<div style="font-family:sans-serif;font-size:12px;color:{INK}">
<table>{rows}
<tr><td style='color:{INK2}'>suggestions</td><td><ol style='margin:0;padding-left:18px'>{sugg_html}</ol></td></tr>
<tr><td style='color:{INK2}'>video</td><td>{rec_row['p1_width']}x{rec_row['p1_height']}, {rec_row['p1_fps_measured']:.3f} fps,
 0.1 s = {rec_row['p1_tolerance_frames']:.1f} frames</td></tr>
<tr><td style='color:{INK2}'>synthetic fields</td><td>{html.escape(str(synth))}</td></tr>
</table></div>"""))
    dur = seg_row.end_s - seg_row.start_s
    times = [seg_row.start_s + dur * (k + 0.5) / n_frames for k in range(n_frames)]
    frames = grab_frames(rec_row["path"], times, width)
    fig, axes = plt.subplots(1, n_frames, figsize=(3.2 * n_frames, 2.1))
    for ax, im, t in zip(np.atleast_1d(axes), frames, times):
        ax.imshow(im)
        ax.set_title(f"{t:.2f} s", fontsize=9, color=INK2)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


# Section 7: primitive outputs -------------------------------------------------

def recording_timeline(prims, recording_id: str, t0: float = 0.0, t1: Optional[float] = None):
    """Segments, hand states and motion signals for one recording on one time axis."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    seg = prims.segments[prims.segments.recording_id == recording_id]
    t1 = t1 if t1 is not None else float(seg.end_s.max())
    seg = seg[(seg.end_s > t0) & (seg.start_s < t1)]
    has_hands = len(prims.hand_timeline) > 0
    has_motion = len(prims.motion) > 0
    n = 1 + (2 if has_hands else 0) + (3 if has_motion else 0)
    ratios = [1.4] + ([0.5, 0.5] if has_hands else []) + ([1.0] * 3 if has_motion else [])
    fig, axes = plt.subplots(n, 1, figsize=(13, 0.55 * sum(ratios) + 1.2), sharex=True,
                             gridspec_kw={"height_ratios": ratios, "hspace": 0.35})
    axes = np.atleast_1d(axes)
    ax = axes[0]
    _style(ax)
    for k, s in enumerate(seg.itertuples()):
        fill = LITERAL_FILL if s.p5_is_fixed_literal else SEG_FILL[k % 2]
        ax.barh(0, s.end_s - s.start_s, left=s.start_s, height=0.8, color=fill, edgecolor=SURFACE, linewidth=2)
        if s.end_s - s.start_s > (t1 - t0) / 60:
            ax.text((s.start_s + s.end_s) / 2, 0, str(s.segment_index), ha="center", va="center",
                    fontsize=7, color=INK)
    ax.set_yticks([0])
    ax.set_yticklabels(["segments"])
    ax.set_title(f"{recording_id}: segments, hand states and motion", loc="left", fontsize=10, color=INK)
    i = 1
    if has_hands:
        ht = prims.hand_timeline[(prims.hand_timeline.recording_id == recording_id)
                                 & (prims.hand_timeline.t >= t0) & (prims.hand_timeline.t <= t1)]
        for side in ("left", "right"):
            ax = axes[i]
            _style(ax)
            h = ht[ht.side == side].sort_values("t")
            dt = np.diff(h.t.to_numpy(), append=h.t.iloc[-1] + 0.1) if len(h) else []
            ax.bar(h.t, 1, width=dt, align="edge", color=[STATE_COLORS[s] for s in h.state], linewidth=0)
            ax.set_yticks([0.5])
            ax.set_yticklabels([f"{side} hand"])
            ax.set_ylim(0, 1)
            i += 1
        names = {"contact": "contact", "no_contact": "no contact", "detected": "detected, no SAM 3 frame near",
                 "unknown": "not detected (unknown)"}
        axes[0].legend(handles=[Patch(color=c, label=names[s]) for s, c in STATE_COLORS.items()],
                       loc="lower right", fontsize=7, ncol=4, frameon=False, bbox_to_anchor=(1, 1.0))
    if has_motion:
        mo = prims.motion[(prims.motion.recording_id == recording_id) & (prims.motion.track == "timeline")
                          & (prims.motion.t >= t0) & (prims.motion.t <= t1)].sort_values("t")
        thr = prims.thresholds.get(recording_id, {})
        ons = prims.onsets[(prims.onsets.recording_id == recording_id)] if len(prims.onsets) else pd.DataFrame()
        labels = {"hand_motion": "hand motion", "camera_turn": "camera turn", "forward": "forward movement"}
        for sig, color in SERIES.items():
            ax = axes[i]
            _style(ax)
            ax.plot(mo.t, mo[sig], color=color, linewidth=1.2)
            if np.isfinite(thr.get(sig, np.nan)):
                ax.axhline(thr[sig], color=MUTED, linewidth=1, linestyle="--")
            if len(ons):
                o = ons[(ons.signal == sig) & (ons.t >= t0) & (ons.t <= t1)]
                ax.plot(o.t, o.value, linestyle="none", marker="o", markersize=4, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=1)
            ax.set_ylabel(labels[sig], rotation=0, ha="right", va="center", fontsize=8)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            i += 1
    for ax in axes:
        for b in seg.start_s:
            ax.axvline(b, color=GRID, linewidth=0.6, zorder=0)
    axes[-1].set_xlim(t0, t1)
    axes[-1].set_xlabel("time (s)")
    if has_motion:
        axes[-1].annotate("Dashed line: onset threshold (median + k x MAD). Dots: onsets.",
                          xy=(0, -0.9), xycoords="axes fraction", fontsize=7, color=INK2)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.9, bottom=0.12)
    plt.show()


def frame_overlay(prims, recording_id: str, frame_idx: int, ax=None):
    """Hand boxes (side) and SAM 3 boxes (prompt) on one sampled frame."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    fr = prims.frames[(prims.frames.recording_id == recording_id) & (prims.frames.frame_idx == frame_idx)]
    if fr.empty:
        raise KeyError(f"frame {frame_idx} of {recording_id} was not sampled")
    im = Image.open(fr.path.iloc[0])
    w, h = im.size
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.imshow(im)
    ax.axis("off")
    ax.set_title(f"{recording_id} frame {frame_idx} at {fr.t.iloc[0]:.2f} s ({fr.segment_id.iloc[0]})",
                 fontsize=8, color=INK2)
    if len(prims.objects):
        ob = prims.objects[(prims.objects.recording_id == recording_id) & (prims.objects.frame_idx == frame_idx)]
        for o in ob.itertuples():
            color = MUTED if o.prompt in ("person",) else SERIES["forward"] if o.prompt != "hand" else SERIES["camera_turn"]
            ax.add_patch(Rectangle((o.x1 * w, o.y1 * h), (o.x2 - o.x1) * w, (o.y2 - o.y1) * h,
                                   fill=False, edgecolor=color, linewidth=1.5))
            ax.text(o.x1 * w, o.y1 * h - 3, f"{o.prompt} {o.score:.2f}", fontsize=7, color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1, alpha=0.85))
    if len(prims.hand_dets):
        hd = prims.hand_dets[(prims.hand_dets.recording_id == recording_id) & (prims.hand_dets.frame_idx == frame_idx)]
        for d in hd.itertuples():
            ax.add_patch(Rectangle((d.x1 * w, d.y1 * h), (d.x2 - d.x1) * w, (d.y2 - d.y1) * h,
                                   fill=False, edgecolor=SERIES["hand_motion"], linewidth=2))
            ax.text(d.x1 * w, d.y2 * h - 4, f"{d.side} hand ({d.owner}) {d.conf:.2f}", fontsize=7, color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1, alpha=0.85))
    if own:
        plt.show()


def cost_summary(cost_per_segment: pd.DataFrame) -> pd.DataFrame:
    """Per primitive: total and per segment seconds and USD."""
    if cost_per_segment.empty:
        return cost_per_segment
    g = cost_per_segment.groupby("name").agg(
        segments=("segment_id", "nunique"),
        total_seconds=("seconds", "sum"),
        median_seconds_per_segment=("seconds", "median"),
        p90_seconds_per_segment=("seconds", lambda s: s.quantile(0.9)),
        api_calls=("api_calls", "sum"),
        total_usd=("total_usd", "sum"),
        median_usd_per_segment=("total_usd", "median"),
    )
    order = [p for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9") if p in g.index]
    return g.loc[order].round(6)


# Section 8: checks -------------------------------------------------------------------

def boundary_strip(prims, recording_id: str, t: float, title: str = "", span: float = 0.6, n: int = 8,
                   marks: Sequence[float] = ()):
    """Sampled frames around a cut. Times are relative to the cut. Frames after it have a blue title."""
    import matplotlib.pyplot as plt
    from PIL import Image

    fr = prims.frames
    f = fr[(fr.recording_id == recording_id) & ((fr.t - t).abs() <= span)].sort_values("t")
    if f.empty:
        return
    pick = set(np.linspace(0, len(f) - 1, min(n, len(f))).round().astype(int))
    pick |= {int(np.argmin((f.t - m).abs().to_numpy())) for m in marks}
    f = f.iloc[sorted(pick)]
    fig, axes = plt.subplots(1, len(f), figsize=(2.3 * len(f), 1.95))
    for ax, r in zip(np.atleast_1d(axes), f.itertuples()):
        ax.imshow(Image.open(r.path))
        ax.axis("off")
        rel = r.t - t
        mark = " *" if any(abs(r.t - m) < 1e-3 for m in marks) else ""
        ax.set_title(f"{rel:+.2f} s{mark}", fontsize=8, color=SERIES["hand_motion"] if rel >= 0 else INK2)
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    plt.show()


def clip_strip(prims, recording_id: str, start: float, end: float, title: str = "", n: int = 8):
    """Timeline frames spread over one segment, with their absolute times."""
    import matplotlib.pyplot as plt
    from PIL import Image

    fr = prims.frames
    f = fr[(fr.recording_id == recording_id) & fr.is_timeline & (fr.t >= start) & (fr.t <= end)].sort_values("t")
    if f.empty:
        return
    f = f.iloc[np.unique(np.linspace(0, len(f) - 1, min(n, len(f))).round().astype(int))]
    fig, axes = plt.subplots(1, len(f), figsize=(2.3 * len(f), 1.95))
    for ax, r in zip(np.atleast_1d(axes), f.itertuples()):
        ax.imshow(Image.open(r.path))
        ax.axis("off")
        ax.set_title(f"{r.t:.1f} s", fontsize=8, color=INK2)
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    plt.show()


def check_report(result, show_flagged: bool = True):
    """Plain language result of one check: the pass count, then every flagged unit with its reason."""
    from IPython.display import Markdown, display

    from .checks.report import check_sentence, flagged_table, unassessable_reasons

    display(Markdown(f"**{check_sentence(result)}**"))
    if show_flagged:
        flagged = flagged_table(result)
        if len(flagged):
            display(Markdown("Flagged for review:"))
            display(flagged)
        reasons = unassessable_reasons(result)
        if len(reasons):
            display(Markdown("Not assessed, by reason:"))
            display(reasons)
