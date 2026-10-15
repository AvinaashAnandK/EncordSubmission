"""P7 segment summary: joins P3, P4, P5 and P6 over each segment.

Most checks read only these columns. Every column is NaN when its source primitive did not run.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .objects import _short
from .schema import ParsedCaption


def _changes(states: pd.Series) -> int:
    s = states[states.isin(["contact", "no_contact"])]
    return int((s != s.shift()).sum() - 1) if len(s) else 0


def _onset_near(rows: pd.DataFrame, t0: float, window: float) -> tuple:
    """First active frame of any signal within the window around t0. (offset, signal)."""
    w = rows[(rows.t >= t0 - window) & (rows.t <= t0 + window)].sort_values("t")
    for r in w.itertuples():
        for s in ("hand_motion", "camera_turn", "forward"):
            if getattr(r, f"{s}_active", False):
                return round(r.t - t0, 3), s
    return np.nan, ""


def segment_summary(seg: pd.DataFrame, parsed: Dict[str, ParsedCaption], frames: pd.DataFrame,
                    dets: pd.DataFrame, hand_tl: pd.DataFrame, motion: pd.DataFrame,
                    objects: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    seg = seg.copy()
    seg["p7_duration_s"] = seg.end_s - seg.start_s
    seg["p7_verb_key"] = [
        next((c.verb_key for c in parsed[s].clauses if not c.is_holding), parsed[s].literal or "")
        for s in seg.segment_id]
    seg["p7_duration_pct_verb"] = seg.groupby("p7_verb_key")["p7_duration_s"].rank(pct=True)

    for r in seg.itertuples():
        d = {"segment_id": r.segment_id, "p7_duration_s": r.p7_duration_s, "p7_verb_key": r.p7_verb_key,
             "p7_duration_pct_verb": r.p7_duration_pct_verb}
        fr = frames[(frames.recording_id == r.recording_id) & (frames.segment_id == r.segment_id)] \
            if len(frames) else frames
        d["p7_n_timeline_frames"] = int(fr.is_timeline.sum()) if len(fr) else np.nan
        d["p7_blur_median"] = float(fr[fr.is_timeline].blur.median()) if len(fr) else np.nan

        if len(hand_tl):
            ht = hand_tl[(hand_tl.recording_id == r.recording_id) & (hand_tl.segment_id == r.segment_id)]
            for side in ("left", "right"):
                h = ht[ht.side == side].sort_values("t")
                n = max(len(h), 1)
                det = h[h.detected]
                d[f"p7_{side}_detected_share"] = round(len(det) / n, 3)
                d[f"p7_{side}_contact_share"] = round((det.state == "contact").mean(), 3) if len(det) else np.nan
                d[f"p7_{side}_contact_changes"] = _changes(h.state)
                objs = h.contact_object.dropna()
                d[f"p7_{side}_contact_object"] = objs.mode().iloc[0] if len(objs) else ""
        if len(dets):
            dd = dets[(dets.recording_id == r.recording_id) & (dets.t >= r.start_s) & (dets.t < r.end_s)]
            # Boxes that do not reach the bottom edge may belong to another person (row 40).
            d["p7_unclear_owner_frames"] = int((dd.owner == "unclear").sum())
        if len(motion):
            m = motion[(motion.recording_id == r.recording_id) & (motion.track == "timeline")
                       & (motion.t > r.start_s) & (motion.t <= r.end_s)]
            if len(m):
                d["p7_forward_share"] = round(float(m.forward_active.mean()), 3)
                d["p7_turn_sum"] = round(float((m.pan_x * m.dt).sum()), 3)
                d["p7_turn_abs_sum"] = round(float((m.pan_x.abs() * m.dt).sum()), 3)
                d["p7_hand_motion_mean"] = round(float(m.hand_motion.mean()), 4)
            near = motion[motion.recording_id == r.recording_id]
            off, sig = _onset_near(near[near.track == "native"] if (near.track == "native").any() else near,
                                   r.start_s, cfg.boundary_window_s)
            d["p7_start_onset_offset_s"], d["p7_start_onset_signal"] = off, sig
        if len(objects):
            ob = objects[objects.segment_id == r.segment_id]
            pc = parsed[r.segment_id]
            main = next((c for c in pc.clauses if not c.is_holding), None)
            target = _short(main.object) if main else ""
            place = _short(main.place) if main else ""
            sam_frames = ob.frame_idx.nunique()
            hit = ob[(ob.prompt == target) & (ob.score >= 0.5)]
            d["p7_object_prompt"] = target
            d["p7_object_found_share"] = round(hit.frame_idx.nunique() / sam_frames, 3) if sam_frames and target else np.nan
            d["p7_object_max_instances"] = int(hit.groupby("frame_idx").obj_id.nunique().max()) if len(hit) else 0
            end_hits = ob[(ob.prompt == place) & (ob.score >= 0.5) & (ob.t >= r.end_s - 1.0)]
            d["p7_place_found_end"] = bool(len(end_hits)) if place else np.nan
            d["p7_person_found"] = bool(((ob.prompt == "person") & (ob.score >= 0.5)).any())
        rows.append(d)
    return pd.DataFrame(rows)
