"""Segmentation errors / Boundary placement: sheet rows 2 to 7.

row02_03  One boundary event, read from both sides. Late cut (row 2): the next action already
          started before the boundary. Early cut (row 3): the earlier action still runs after it.
row04     Possible filler merge: a boundary flag with a shift of 0.1 s to 0.2 s and a short motion.
row05     Export gate: contract, contiguity, minimum length, coverage of the video.
row06     Reach start vs earliest visible evidence (a measurement, not a verdict).
row07     A split where the same action resumes after a short gap (object briefly out of view).
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import lexicon as L
from ..clients import Question
from ..schema import Finding
from .runner import CheckContext, CheckSpec, Unit, register

CATEGORY = "Segmentation errors / Boundary placement"
REACH_VERBS = {"pick up", "grab", "take", "grasp", "reach", "lift", "pick"}


# Shared helpers ------------------------------------------------------------------

def boundary_units(ctx: CheckContext) -> Iterable[Unit]:
    seg = ctx.prims.segments.sort_values(["recording_id", "segment_index"])
    for rid, g in seg.groupby("recording_id"):
        rows = g.to_dict("records")
        for prev, nxt in zip(rows, rows[1:]):
            yield {"recording_id": rid, "segment_id": nxt["segment_id"], "prev_id": prev["segment_id"],
                   "t": float(nxt["start_s"]), "prev_caption": prev["caption"], "next_caption": nxt["caption"]}


def segment_units(ctx: CheckContext) -> Iterable[Unit]:
    for r in ctx.prims.segments.sort_values(["recording_id", "segment_index"]).itertuples():
        yield {"recording_id": r.recording_id, "segment_id": r.segment_id}


def _main_clause(ctx, sid):
    pc = ctx.prims.parsed[sid]
    return next((c for c in pc.clauses if not c.is_holding), None)


def _sides(ctx, sid) -> List[str]:
    c = _main_clause(ctx, sid)
    actor = c.actor if c else ""
    if actor == "right hand":
        return ["right"]
    if actor == "left hand":
        return ["left"]
    return ["left", "right"]


def _describe(caption: str, role: str) -> str:
    """Plain description of a caption for an image question."""
    lit = caption.strip()
    if lit == "IDLE":
        return "the person is not acting (IDLE)"
    if lit == "SITTING":
        return "the person is seated (SITTING)"
    if lit == "TASK UNRELATED ACTION":
        return "the person does something unrelated to the task"
    return f'"{caption}"'


def _frames_near(ctx, rid: str, t: float, window: float) -> pd.DataFrame:
    fr = ctx.prims.frames
    f = fr[(fr.recording_id == rid) & ((fr.t - t).abs() <= window)]
    return f.sort_values("t")


def _rising(times: np.ndarray, active: np.ndarray) -> List[float]:
    prev = np.r_[False, active[:-1]]
    return [float(t) for t in times[active & ~prev]]


def _signal_candidates(ctx, rid: str, t: float, sides: List[str]) -> Tuple[List[Tuple[float, str]], bool]:
    """Times in the window where something visibly starts. Returns (candidates, signals_present)."""
    w = ctx.cfg.boundary_window_s
    cands: List[Tuple[float, str]] = []
    present = False
    mo = ctx.prims.motion
    if len(mo):
        m = mo[(mo.recording_id == rid) & ((mo.t - t).abs() <= w)]
        m = m[m.track == "native"] if (m.track == "native").any() else m[m.track == "timeline"]
        m = m.sort_values("t")
        if len(m):
            present = True
            for s in ("hand_motion", "camera_turn", "forward"):
                cands += [(x, s) for x in _rising(m.t.to_numpy(), m[f"{s}_active"].to_numpy(bool))]
    ht = ctx.prims.hand_timeline
    if len(ht):
        h = ht[(ht.recording_id == rid) & ((ht.t - t).abs() <= w) & ht.side.isin(sides)].sort_values("t")
        if len(h):
            present = True
            for side, g in h.groupby("side"):
                g = g[g.state.isin(["contact", "no_contact"])]
                change = g.state != g.state.shift()
                for r in g[change].iloc[1:].itertuples():
                    cands.append((float(r.t), f"{side}_{r.state}"))
    return sorted(cands), present


# Rows 2 and 3: one boundary event --------------------------------------------------

def _run_row02_03(ctx: CheckContext, u: Unit) -> List[Finding]:
    cfg = ctx.cfg
    tol = cfg.boundary_tolerance_effective_s
    p_flag = cfg.image_flag_prob
    rid, t = u["recording_id"], u["t"]
    cands, signals = _signal_candidates(ctx, rid, t, _sides(ctx, u["segment_id"]))
    nearest = min(cands, key=lambda c: abs(c[0] - t)) if cands else None
    offset = round(nearest[0] - t, 3) if nearest else None

    # Ask about the frames at the edges of the tolerance, not next to the cut. A frame 40 ms
    # after a cut can show the earlier action and still be within tolerance.
    near = _frames_near(ctx, rid, t, cfg.boundary_window_s)
    before = near[near.t <= t - tol + 1e-6].tail(1)
    after = near[near.t >= t + tol - 1e-6].head(1)
    # One comparison per boundary: the frame at cut - tolerance and the frame at cut + tolerance.
    # If the cut is right, the change from the earlier action to the next one happens between them.
    change = None
    if len(before) and len(after):
        q = Question(
            "change_when", "choice",
            "Two frames from a first person video, in time order. The person first does "
            f"{_describe(u['prev_caption'], 'prev')} and then does {_describe(u['next_caption'], 'next')}. "
            "When does the person switch from the first action to the second?",
            {"before_first": "The second action is already under way in the first frame.",
             "between": "The first frame still shows the first action and the second frame shows the second action starting.",
             "after_second": "The second frame still shows the first action. The switch comes later.",
             "unclear": "The frames do not show when the switch happens."})
        change = ctx.clients.decisions.ask(
            [q], f"Frame 1 is at {before.t.iloc[0] - t:+.2f} s from the cut. Frame 2 is at {after.t.iloc[0] - t:+.2f} s.",
            [before.path.iloc[0], after.path.iloc[0]])[q.name]
    probs = (change.probs or {}) if change is not None and change.status == "ok" else {}
    late = SimpleNamespace(status="ok" if probs else "unassessable", prob=probs.get("before_first"),
                           source=getattr(change, "source", ""))
    early = SimpleNamespace(status="ok" if probs else "unassessable", prob=probs.get("after_second"),
                            source=getattr(change, "source", ""))
    late_img = bool(probs) and (late.prob or 0) >= p_flag
    early_img = bool(probs) and (early.prob or 0) >= p_flag
    late_sig = offset is not None and offset < -tol
    early_sig = offset is not None and offset > tol
    no_trigger = signals and not cands
    images_ok = bool(probs)
    rng = random.Random(f"{cfg.seed}:{u['segment_id']}")
    ev = {
        "t": t, "prev_id": u["prev_id"], "tolerance_s": tol, "signals_present": signals,
        "offset_s": offset, "trigger": nearest[1] if nearest else "", "n_candidates": len(cands),
        "late_prob": late.prob, "early_prob": early.prob, "between_prob": probs.get("between"),
        "unclear_prob": probs.get("unclear"), "image_source": late.source,
        "frame_before": int(before.frame_idx.iloc[0]) if len(before) else None,
        "t_before": float(before.t.iloc[0]) if len(before) else None,
        "frame_after": int(after.frame_idx.iloc[0]) if len(after) else None,
        "t_after": float(after.t.iloc[0]) if len(after) else None,
        "audit_sample": rng.random() < cfg.boundary_audit_rate,
    }
    base = dict(check="row02_03", recording_id=rid, segment_id=u["segment_id"], evidence=ev)
    if not signals and not images_ok:
        return [Finding(row="2, 3", status="unassessable",
                        message="No motion or hand signal and no image answer for this boundary.", **base)]
    is_late, is_early = late_sig or late_img, early_sig or early_img
    if is_late and is_early:
        return [Finding(row="2, 3", status="candidate", message="Signals disagree on the direction. Review.", **base)]
    if is_late:
        why = [w for w, on in ((f"trigger {offset:+.2f} s" if offset is not None else "", late_sig),
                               (f"switch already under way {tol:.2f} s before the cut (p={ev['late_prob']:.2f})", late_img)) if on]
        return [Finding(row="2", status="candidate", message="Late cut: " + "; ".join(why), **base)]
    if is_early:
        why = [w for w, on in ((f"trigger {offset:+.2f} s" if offset is not None else "", early_sig),
                               (f"earlier action still running {tol:.2f} s after the cut (p={ev['early_prob']:.2f})", early_img)) if on]
        return [Finding(row="3", status="candidate", message="Early cut: " + "; ".join(why), **base)]
    if no_trigger:
        return [Finding(row="2, 3", status="candidate",
                        message="No visible trigger within the window. A large error can sit outside it.", **base)]
    return [Finding(row="2, 3", status="pass", message="Cut agrees with the evidence.", **base)]


register(CheckSpec(
    key="row02_03", rows="2, 3", title="Boundary event: late or early cut", category=CATEGORY,
    uses=["P2", "P3", "P4"], units=boundary_units, run=_run_row02_03, workers=8, unit="boundary"))


# Row 4: possible filler merge --------------------------------------------------------

def _run_row04(ctx: CheckContext, u: Unit) -> List[Finding]:
    import json

    base = dict(row="4", check="row04", recording_id=u["recording_id"], segment_id=u["segment_id"])
    prev = ctx.results.get("row02_03")
    if prev is None:
        return [Finding(status="unassessable", message="Run row02_03 first.", **base)]
    hit = prev[prev.segment_id == u["segment_id"]]
    if hit.empty:
        return [Finding(status="unassessable", message="No boundary finding.", **base)]
    ev = json.loads(hit.evidence.iloc[0])
    offset = ev.get("offset_s")
    if hit.status.iloc[0] != "candidate":
        return [Finding(status="pass", message="No boundary flag here.", evidence={"offset_s": offset}, **base)]
    if offset is None or not len(ctx.prims.motion):
        return [Finding(status="unassessable", message="Needs the motion timeline (P4) for the shift and the burst.",
                        evidence={"offset_s": offset}, **base)]
    if not (0.1 <= abs(offset) <= 0.2):
        return [Finding(status="pass", message="Shift outside the 0.1 s to 0.2 s band.", evidence={"offset_s": offset}, **base)]
    mo = ctx.prims.motion
    m = mo[(mo.recording_id == u["recording_id"]) & ((mo.t - u["t"]).abs() <= 0.25) & (mo.track == "native")].sort_values("t")
    act = m[["hand_motion_active", "camera_turn_active", "forward_active"]].any(axis=1).to_numpy() if len(m) else np.array([])
    runs, i = [], 0
    while i < len(act):
        if act[i]:
            j = i
            while j < len(act) and act[j]:
                j += 1
            runs.append(float(m.t.iloc[j - 1] - m.t.iloc[i]))
            i = j
        else:
            i += 1
    short = [r for r in runs if r < ctx.cfg.min_segment_s]
    evd = {"offset_s": offset, "burst_lengths_s": runs}
    if short:
        return [Finding(status="candidate", message="Possible filler merge: a motion under 0.2 s at the boundary.",
                        evidence=evd, **base)]
    return [Finding(status="report", message="Shift in the 0.1 s to 0.2 s band, but no short motion visible.",
                    evidence=evd, **base)]


register(CheckSpec(
    key="row04", rows="4", title="Possible filler merge", category=CATEGORY,
    uses=["P4"], units=boundary_units, run=_run_row04, unit="boundary"))


# Row 5: export gate -----------------------------------------------------------------

def _run_row05(ctx: CheckContext, u: Unit) -> List[Finding]:
    cfg = ctx.cfg
    eps = cfg.gate_epsilon_s
    seg = ctx.prims.segments
    g = seg[seg.recording_id == u["recording_id"]].sort_values("start_s").reset_index(drop=True)
    i = int(g.index[g.segment_id == u["segment_id"]][0])
    r = g.iloc[i]
    rec = ctx.prims.recordings.set_index("recording_id").loc[u["recording_id"]]
    duration, frame = float(rec.p1_duration_s), 1.0 / float(rec.p1_fps_measured)
    fails: List[str] = []
    if not str(r.caption).strip():
        fails.append("blank caption")
    if not (math.isfinite(r.start_s) and math.isfinite(r.end_s)):
        fails.append("time is not a finite number")
    if (g.segment_id == r.segment_id).sum() > 1:
        fails.append("duplicate segment id")
    if i > 0 and r.segment_index <= g.segment_index.iloc[i - 1]:
        fails.append("segment numbers out of order")
    if r.start_s < -eps or r.end_s > duration + frame:
        fails.append(f"time outside the video (0 to {duration:.3f} s)")
    if i + 1 < len(g):
        gap = float(g.start_s.iloc[i + 1] - r.end_s)
        if gap > eps:
            fails.append(f"gap of {gap:.3f} s before the next segment")
        elif gap < -eps:
            fails.append(f"overlap of {-gap:.3f} s with the next segment")
    if r.end_s - r.start_s < cfg.min_segment_s - eps:
        fails.append(f"shorter than {cfg.min_segment_s} s")
    if i == 0 and abs(r.start_s) > eps:
        fails.append(f"first segment starts at {r.start_s:.3f} s, not 0")
    if i == len(g) - 1 and abs(r.end_s - duration) > frame:
        fails.append(f"last segment ends at {r.end_s:.3f} s but the video ends at {duration:.3f} s")
    ev = {"start_s": float(r.start_s), "end_s": float(r.end_s), "video_s": duration, "one_frame_s": round(frame, 4)}
    base = dict(row="5", check="row05", recording_id=u["recording_id"], segment_id=u["segment_id"], evidence=ev)
    if fails:
        return [Finding(status="candidate", message="Gate failure: " + "; ".join(fails), **base)]
    return [Finding(status="pass", message="", **base)]


register(CheckSpec(
    key="row05", rows="5", title="Export gate", category=CATEGORY,
    uses=["P1"], units=segment_units, run=_run_row05))


# Row 6: reach start vs earliest visible evidence --------------------------------------

def _reach_units(ctx: CheckContext) -> Iterable[Unit]:
    for u in segment_units(ctx):
        c = _main_clause(ctx, u["segment_id"])
        if c is not None and c.verb_key in REACH_VERBS:
            yield u


def _run_row06(ctx: CheckContext, u: Unit) -> List[Finding]:
    seg = ctx.prims.segments.set_index("segment_id").loc[u["segment_id"]]
    start = float(seg.start_s)
    base = dict(row="6", check="row06", recording_id=u["recording_id"], segment_id=u["segment_id"])
    ht, mo = ctx.prims.hand_timeline, ctx.prims.motion
    if not len(ht):
        return [Finding(status="unassessable", message="Needs the hand timeline (P3).", **base)]
    sides = _sides(ctx, u["segment_id"])
    lo, hi = start - 1.0, start + 0.5
    times: List[Tuple[float, str]] = []
    h = ht[(ht.recording_id == u["recording_id"]) & ht.side.isin(sides) & (ht.t >= lo) & (ht.t <= hi)].sort_values("t")
    for side, g in h.groupby("side"):
        det = g.detected.to_numpy(bool)
        times += [(x, f"{side} hand enters") for x in _rising(g.t.to_numpy(), det) if x > lo]
    if len(mo):
        m = mo[(mo.recording_id == u["recording_id"]) & (mo.t >= lo) & (mo.t <= hi)
               & (mo.track == "timeline")].sort_values("t")
        thr = ctx.prims.thresholds.get(u["recording_id"], {}).get("hand_motion", np.nan)
        for side in sides:
            col = f"hand_motion_{side}"
            if col in m and np.isfinite(thr):
                times += [(x, f"{side} hand motion starts") for x in _rising(m.t.to_numpy(), (m[col] > thr).to_numpy())
                          if x > lo]
    if not times:
        return [Finding(status="unassessable", message="No visible start of the reach in the window.",
                        evidence={"start_s": start}, **base)]
    first = min(times)
    gap = round(start - first[0], 3)
    ev = {"start_s": start, "evidence_t": first[0], "evidence": first[1], "gap_s": gap}
    return [Finding(status="report", message=f"Start sits {gap:+.2f} s after the earliest visible evidence ({first[1]}).",
                    evidence=ev, **base)]


register(CheckSpec(
    key="row06", rows="6", title="Reach start vs earliest visible evidence", category=CATEGORY,
    uses=["P3", "P4", "P5"], units=_reach_units, run=_run_row06))


# Row 7: split where the same action resumes ---------------------------------------------

def _middle_units(ctx: CheckContext) -> Iterable[Unit]:
    seg = ctx.prims.segments.sort_values(["recording_id", "segment_index"])
    for rid, g in seg.groupby("recording_id"):
        rows = g.to_dict("records")
        for a, b, c in zip(rows, rows[1:], rows[2:]):
            yield {"recording_id": rid, "segment_id": b["segment_id"], "s1": a["segment_id"], "s3": c["segment_id"],
                   "s2_duration": float(b["end_s"] - b["start_s"]), "s1_end": float(a["end_s"]),
                   "s3_start": float(c["start_s"])}


def _same_action(c1, c3) -> bool:
    if c1 is None or c3 is None or not c1.verb_key or c1.verb_key != c3.verb_key:
        return False
    if frozenset((c1.verb_key, c3.verb_key)) in L.OPPOSITE_VERBS:
        return False
    if c1.object and c1.object == c3.object:
        return True
    return bool(c1.object_head) and c1.object_head == c3.object_head and not (
        set(c1.attributes) ^ set(c3.attributes)) - set(L.FALLBACK_NOUNS)


def _run_row07(ctx: CheckContext, u: Unit) -> List[Finding]:
    base = dict(row="7", check="row07", recording_id=u["recording_id"], segment_id=u["segment_id"])
    c1, c3 = _main_clause(ctx, u["s1"]), _main_clause(ctx, u["s3"])
    if u["s2_duration"] >= 1.0 or not _same_action(c1, c3):
        return [Finding(status="pass", message="No same action on both sides of a short segment.", **base)]
    ev = {"s1": u["s1"], "s3": u["s3"], "verb": c1.verb_key, "object": c1.object, "s2_duration_s": u["s2_duration"]}
    ht = ctx.prims.hand_timeline
    if not len(ht):
        return [Finding(status="unassessable", message="Text match. Needs the hand timeline (P3) to confirm.",
                        evidence=ev, **base)]
    sides = _sides(ctx, u["s1"])
    h = ht[(ht.recording_id == u["recording_id"]) & ht.side.isin(sides)]
    end1 = h[(h.t <= u["s1_end"]) & (h.t > u["s1_end"] - 0.5)]
    start3 = h[(h.t >= u["s3_start"]) & (h.t < u["s3_start"] + 0.5)]
    obj = (c1.object or "").split(" of ")[0]
    held1 = (end1.contact_object == obj).any()
    held3 = (start3.contact_object == obj).any()
    ev.update({"held_before": bool(held1), "held_after": bool(held3)})
    if held1 and held3:
        return [Finding(status="candidate", message=f"Same {obj} in the same hand before and after a "
                        f"{u['s2_duration']:.1f} s segment. Review the split.", evidence=ev, **base)]
    return [Finding(status="pass", message="Text match, but the frames do not show the same object held throughout.",
                    evidence=ev, **base)]


register(CheckSpec(
    key="row07", rows="7", title="Split where the same action resumes", category=CATEGORY,
    uses=["P3", "P5", "P6"], units=_middle_units, run=_run_row07))


BOUNDARY_CHECKS = ["row05", "row02_03", "row04", "row06", "row07"]
