"""Segmentation errors / Non-action states: sheet rows 18 to 27.

row18  A pause of 1 s or more left inside an action segment (a missed IDLE).
row19  An IDLE under 1 s (allowed: reported, never scored).
row20  Hand actions captioned while the person sits (they belong to SITTING).
row21  SITTING that does not cover sitting down or standing up.
row22  Kneeling or squatting (a guideline gap: a policy queue, never scored).
row23  A clear task unrelated action marked IDLE.
row24  An identifiable action marked IDLE.
row25  A task action marked TASK UNRELATED ACTION.
row26  Looking around (IDLE) confused with a turn (navigation).
row27  A non-action stretch missed entirely inside a long segment.

Frame signals come from P3 and P4 when they ran. Without them, OpenAI Decisions answers closed
questions on sampled frames. Rows that ask the same question (posture for rows 20 to 22, "what is
the person doing" for rows 23 and 24) send it in exactly the same form, so the later check reads the
cached answer.
"""
from __future__ import annotations

import json
import random
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .. import lexicon as L
from ..clients import Question
from ..schema import Finding
from .boundary import _main_clause, segment_units
from .granularity import _f, _prob, _rng, _seg, clip_frames, frame_at
from .runner import CheckContext, CheckSpec, Unit, register

CATEGORY = "Segmentation errors / Non-action states"

POSTURE = {"standing": "The camera wearer stands.",
           "seated": "The camera wearer sits: the view is low, as from a chair or sofa, and the lap or a nearby table edge is close.",
           "kneeling_or_squatting": "The camera wearer kneels or squats: the view is very low, close to the floor or a low cupboard.",
           "unclear": "The frames do not show the posture."}
DOING = {"unrelated_action": "Something not part of the task: waving or talking to someone, adjusting a headset or camera, or preparing the scene.",
         "task_action": "Actively moving, picking up, placing or working on an object that is part of the task.",
         "walking": "Walking or moving to another place.",
         "not_acting": "Standing or sitting still, or only looking around. Hands that rest on an object without moving it count here.",
         "unclear": "The frames do not show what the person does."}


def _caption(ctx, sid) -> str:
    return str(_seg(ctx, sid).caption).strip()


def _literal(ctx, sid) -> str:
    pc = ctx.prims.parsed[sid]
    return pc.literal if pc.is_fixed_literal else ""


def _units_where(ctx, pred) -> Iterable[Unit]:
    for u in segment_units(ctx):
        if pred(ctx, u["segment_id"]):
            yield u


def _ask_frames(ctx, q: Question, rid: str, start: float, end: float, context: str, n: int):
    paths = clip_frames(ctx, rid, start, end, n)
    if not paths:
        return None, []
    state = (f"{context}\nThe {len(paths)} images are frames of a first person video, in time order, "
             f"from {start:.1f} s to {end:.1f} s.")
    return ctx.clients.decisions.ask([q], state, paths)[q.name], paths


def _posture(ctx, sid):
    """Shared by rows 20 and 22: posture over up to 3 frames of the segment."""
    s = _seg(ctx, sid)
    q = Question("posture", "choice", "What is the posture of the camera wearer in these frames?", POSTURE)
    return _ask_frames(ctx, q, s.recording_id, s.start_s, s.end_s, "", 3)[0]


def _doing(ctx, sid):
    """Shared by rows 23 and 24: what the person does during an IDLE segment.

    The frames come from the middle of the segment. Frames at its edges show the neighbouring
    actions and bias the answer toward "task action".
    """
    s = _seg(ctx, sid)
    d = float(s.end_s - s.start_s)
    q = Question("doing", "choice", "What is the camera wearer doing in these frames?", DOING)
    return _ask_frames(ctx, q, s.recording_id, s.start_s + 0.25 * d, s.end_s - 0.25 * d, "", 3)[0]


# Row 18: a pause left inside an action segment -------------------------------------------

def _row18_units(ctx):
    return _units_where(ctx, lambda c, sid: not _literal(c, sid) and float(_seg(c, sid).end_s - _seg(c, sid).start_s) > 1.0)


def _still_stretches(ctx, rid, start, end) -> Optional[List[tuple]]:
    """Stretches of 1 s or more with both hands seen, no contact, no hand motion, no walking (P3, P4)."""
    ht, mo = ctx.prims.hand_timeline, ctx.prims.motion
    if not len(ht) or not len(mo):
        return None
    h = ht[(ht.recording_id == rid) & (ht.t >= start) & (ht.t < end)]
    m = mo[(mo.recording_id == rid) & (mo.track == "timeline") & (mo.t >= start) & (mo.t < end)].set_index("frame_idx")
    piv = h.pivot_table(index="frame_idx", columns="side", values="state", aggfunc="first")
    t = h.groupby("frame_idx").t.first()
    still = ((piv == "no_contact").all(axis=1)
             & ~m.reindex(piv.index).hand_motion_active.fillna(True).astype(bool)
             & ~m.reindex(piv.index).forward_active.fillna(True).astype(bool))
    runs, cur = [], []
    for fi in piv.index:
        if still.get(fi, False):
            cur.append(float(t[fi]))
        else:
            if cur and cur[-1] - cur[0] >= 1.0:
                runs.append((cur[0], cur[-1]))
            cur = []
    if cur and cur[-1] - cur[0] >= 1.0:
        runs.append((cur[0], cur[-1]))
    return runs


def _run_row18(ctx: CheckContext, u: Unit) -> List[Finding]:
    s = _seg(ctx, u["segment_id"])
    runs = _still_stretches(ctx, u["recording_id"], float(s.start_s), float(s.end_s))
    ev = {"duration_s": float(s.end_s - s.start_s), "still_stretches": runs}
    if runs:
        return [_f("18", "row18", u, "candidate",
                   f"Possible missed IDLE: {len(runs)} still stretch(es) of 1 s or more, first at {runs[0][0]:.1f} s.", ev)]
    if runs is not None:
        return [_f("18", "row18", u, "pass", "No still stretch of 1 s or more.", ev)]
    # No GPU timelines: ask about the frames instead.
    q = Question("pause", "noul",
                 "Is there a stretch of about one second or more in this clip where the camera wearer pauses: the "
                 "hands are idle and the person is not doing the task (looking around counts as a pause)?")
    ans, paths = _ask_frames(ctx, q, u["recording_id"], float(s.start_s), float(s.end_s),
                             f'The caption of this clip is "{s.caption}".', 12)
    p = _prob(ans)
    ev.update({"pause_prob": p, "frames": len(paths), "source": getattr(ans, "source", "")})
    if p is None:
        return [_f("18", "row18", u, "unassessable", "No timelines and no frame answer.", ev)]
    if p >= ctx.cfg.image_flag_prob:
        return [_f("18", "row18", u, "candidate",
                   f"Possible missed IDLE: the frames show a pause inside the segment (p={p:.2f}).", ev)]
    return [_f("18", "row18", u, "pass", "No pause in the sampled frames.", ev)]


register(CheckSpec(key="row18", rows="18", title="Pause left inside an action segment", category=CATEGORY,
                   uses=["P2", "P3", "P4"], units=_row18_units, run=_run_row18, workers=8))


# Row 19: IDLE under 1 s (allowed) ----------------------------------------------------------

def _idle_units(ctx):
    return _units_where(ctx, lambda c, sid: _literal(c, sid) == "IDLE")


def _run_row19(ctx: CheckContext, u: Unit) -> List[Finding]:
    s = _seg(ctx, u["segment_id"])
    d = round(float(s.end_s - s.start_s), 3)
    ev = {"duration_s": d}
    if d < ctx.cfg.idle_pause_s:
        note = " Within the 0.1 s rounding of the CSV." if d >= ctx.cfg.idle_pause_s - ctx.cfg.csv_time_resolution_s else ""
        return [_f("19", "row19", u, "report", f"IDLE of {d:.1f} s, under 1 s. Allowed by the guideline, not scored.{note}", ev)]
    return [_f("19", "row19", u, "pass", f"IDLE of {d:.1f} s.", ev)]


register(CheckSpec(key="row19", rows="19", title="IDLE under 1 s", category=CATEGORY,
                   uses=[], units=_idle_units, run=_run_row19))


# Row 20: seated hand actions captioned -------------------------------------------------------

def _hand_action_units(ctx):
    return _units_where(ctx, lambda c, sid: not _literal(c, sid))


def _run_row20(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    g = ctx.prims.segments[ctx.prims.segments.recording_id == u["recording_id"]].sort_values("segment_index")
    lits = [ctx.prims.parsed[x].literal if ctx.prims.parsed[x].is_fixed_literal else "" for x in g.segment_id]
    i = list(g.segment_id).index(sid)
    near_sitting = "SITTING" in lits[max(0, i - 1):i + 2]
    ans = _posture(ctx, sid)
    probs = (ans.probs or {}) if ans is not None and ans.status == "ok" else {}
    ev = {"near_sitting": near_sitting, "seated_prob": probs.get("seated"), "posture": getattr(ans, "value", None),
          "source": getattr(ans, "source", "")}
    if not probs:
        return [_f("20", "row20", u, "unassessable", "No posture answer.", ev)]
    if probs.get("seated", 0) >= ctx.cfg.image_flag_prob:
        return [_f("20", "row20", u, "candidate",
                   f"Captioned as an action while the person sits (p={probs['seated']:.2f}). Seated actions belong to SITTING.", ev)]
    return [_f("20", "row20", u, "pass", f"Posture: {ev['posture']}.", ev)]


register(CheckSpec(key="row20", rows="20", title="Seated hand actions captioned", category=CATEGORY,
                   uses=["P2", "P4", "P5"], units=_hand_action_units, run=_run_row20, workers=8))


# Row 21: SITTING does not cover sitting down or standing up ------------------------------------

def _sitting_units(ctx):
    return _units_where(ctx, lambda c, sid: _literal(c, sid) == "SITTING")


def _run_row21(ctx: CheckContext, u: Unit) -> List[Finding]:
    s = _seg(ctx, u["segment_id"])
    rid = u["recording_id"]
    q = Question("posture_one", "choice", "What is the posture of the camera wearer in this frame?",
                 {**POSTURE, "changing": "The camera wearer is sitting down or standing up."})
    out = {}
    for name, t in (("before_start", float(s.start_s) - 0.3), ("after_end", float(s.end_s) + 0.3)):
        p = frame_at(ctx, rid, t)
        a = ctx.clients.decisions.ask([q], "One frame from a first person video.", [p])[q.name] if p else None
        out[name] = (a.probs or {}) if a is not None and a.status == "ok" else {}
    ev = {k: v for k, v in out.items()}
    if not out["before_start"] and not out["after_end"]:
        return [_f("21", "row21", u, "unassessable", "No posture answer around the SITTING segment.", ev)]
    msgs = []
    if out["before_start"].get("seated", 0) + out["before_start"].get("changing", 0) >= ctx.cfg.image_flag_prob:
        msgs.append("the person is already sitting or sitting down before SITTING starts")
    if out["after_end"].get("seated", 0) + out["after_end"].get("changing", 0) >= ctx.cfg.image_flag_prob:
        msgs.append("the person is still seated or standing up after SITTING ends")
    if msgs:
        return [_f("21", "row21", u, "candidate", "SITTING misses part of the transition: " + "; ".join(msgs) + ".", ev)]
    return [_f("21", "row21", u, "pass", "SITTING covers sitting down and standing up.", ev)]


register(CheckSpec(key="row21", rows="21", title="SITTING misses sitting down or standing up", category=CATEGORY,
                   uses=["P2", "P4"], units=_sitting_units, run=_run_row21, workers=8))


# Row 22: kneeling or squatting (policy queue) -----------------------------------------------------

def _run_row22(ctx: CheckContext, u: Unit) -> List[Finding]:
    ans = _posture(ctx, u["segment_id"])
    probs = (ans.probs or {}) if ans is not None and ans.status == "ok" else {}
    ev = {"kneel_squat_prob": probs.get("kneeling_or_squatting"), "posture": getattr(ans, "value", None)}
    if not probs:
        return [_f("22", "row22", u, "unassessable", "No posture answer.", ev)]
    if probs.get("kneeling_or_squatting", 0) >= ctx.cfg.image_flag_prob:
        return [_f("22", "row22", u, "report", f"Kneeling or squatting (p={ev['kneel_squat_prob']:.2f}). Posture "
                   f"unclear in the guideline: sent to the policy queue, not scored. Caption: \"{_caption(ctx, u['segment_id'])}\".", ev)]
    return [_f("22", "row22", u, "pass", f"Posture: {ev['posture']}.", ev)]


register(CheckSpec(key="row22", rows="22", title="Kneeling or squatting (policy queue)", category=CATEGORY,
                   uses=["P2"], units=_hand_action_units, run=_run_row22, workers=8))


# Rows 23 and 24: what happens inside IDLE ------------------------------------------------------

def _run_row23(ctx: CheckContext, u: Unit) -> List[Finding]:
    s = _seg(ctx, u["segment_id"])
    d = float(s.end_s - s.start_s)
    if d <= 3.0:
        return [_f("23", "row23", u, "pass", f"IDLE of {d:.1f} s, too short for this check (over 3 s).", {"duration_s": d})]
    ans = _doing(ctx, u["segment_id"])
    probs = (ans.probs or {}) if ans is not None and ans.status == "ok" else {}
    ev = {"duration_s": d, "unrelated_prob": probs.get("unrelated_action"), "answer": getattr(ans, "value", None)}
    if len(ctx.prims.hand_dets):
        dd = ctx.prims.hand_dets
        ev["unclear_owner_frames"] = int(((dd.recording_id == u["recording_id"]) & (dd.t >= s.start_s)
                                          & (dd.t < s.end_s) & (dd.owner == "unclear")).sum())
    if not probs:
        return [_f("23", "row23", u, "unassessable", "No frame answer.", ev)]
    p = probs.get("unrelated_action", 0)
    if p >= ctx.cfg.image_flag_prob:
        return [_f("23", "row23", u, "candidate", f"A clear task unrelated action marked IDLE (p={p:.2f}).", ev)]
    if p >= 1 - ctx.cfg.image_flag_prob:
        return [_f("23", "row23", u, "report", f"Maybe a task unrelated action (p={p:.2f}). Unclear case, not scored.", ev)]
    return [_f("23", "row23", u, "pass", f"No unrelated action ({ev['answer']}).", ev)]


register(CheckSpec(key="row23", rows="23", title="Task unrelated action marked IDLE", category=CATEGORY,
                   uses=["P2", "P3", "P6"], units=_idle_units, run=_run_row23, workers=8))


def _run_row24(ctx: CheckContext, u: Unit) -> List[Finding]:
    ans = _doing(ctx, u["segment_id"])
    probs = (ans.probs or {}) if ans is not None and ans.status == "ok" else {}
    acting = probs.get("task_action", 0) + probs.get("walking", 0) if probs else None
    ev = {"acting_prob": acting, "answer": getattr(ans, "value", None)}
    s = _seg(ctx, u["segment_id"])
    if len(ctx.prims.motion):
        mo = ctx.prims.motion
        m = mo[(mo.recording_id == u["recording_id"]) & (mo.track == "timeline") & (mo.t > s.start_s) & (mo.t <= s.end_s)]
        ev["forward_share"] = float(m.forward_active.mean()) if len(m) else None
    if acting is None:
        return [_f("24", "row24", u, "unassessable", "No frame answer.", ev)]
    if acting >= ctx.cfg.image_flag_prob:
        what = "walking" if probs.get("walking", 0) > probs.get("task_action", 0) else "a task action"
        return [_f("24", "row24", u, "candidate", f"Marked IDLE but the frames show {what} (p={acting:.2f}). "
                   "The reviewer writes the caption if the action is identifiable.", ev)]
    return [_f("24", "row24", u, "pass", f"The frames agree with IDLE ({ev['answer']}).", ev)]


register(CheckSpec(key="row24", rows="24", title="Identifiable action marked IDLE", category=CATEGORY,
                   uses=["P2", "P3", "P4"], units=_idle_units, run=_run_row24, workers=8))


# Row 25: task action marked TASK UNRELATED ACTION ----------------------------------------------

def _tua_units(ctx):
    return _units_where(ctx, lambda c, sid: _literal(c, sid) == "TASK UNRELATED ACTION")


TASK_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["task", "confidence"],
               "properties": {"task": {"type": "string"}, "confidence": {"type": "number"}}}


def _task_of(ctx, rid) -> dict:
    caps = [c for c in ctx.prims.segments[ctx.prims.segments.recording_id == rid].sort_values("segment_index").caption
            if c not in L.FIXED_LITERALS]
    data = ctx.clients.sol.json("These captions describe one first person video in order. State the overall task "
                                "in one short line, and your confidence from 0 to 1.\n" + "\n".join(caps),
                                TASK_SCHEMA, name="task")
    return data or {}


def _run_row25(ctx: CheckContext, u: Unit) -> List[Finding]:
    task = _task_of(ctx, u["recording_id"])
    ev = {"task": task.get("task"), "task_confidence": task.get("confidence")}
    if not task or float(task.get("confidence", 0)) < 0.5:
        return [_f("25", "row25", u, "unassessable", "The task could not be inferred from the captions.", ev)]
    s = _seg(ctx, u["segment_id"])
    q = Question("task_step", "noul", f'The task is: "{task["task"]}". Do these frames show a step of that task? '
                 "Talking or waving to someone, adjusting a headset or preparing the scene are not task steps.")
    ans, _ = _ask_frames(ctx, q, u["recording_id"], float(s.start_s), float(s.end_s), "", 3)
    p = _prob(ans)
    ev["task_step_prob"] = p
    if p is None:
        return [_f("25", "row25", u, "unassessable", "No frame answer.", ev)]
    if p >= ctx.cfg.image_flag_prob:
        return [_f("25", "row25", u, "candidate", f"Marked TASK UNRELATED ACTION but looks like a step of the task (p={p:.2f}).", ev)]
    return [_f("25", "row25", u, "pass", "Not a task step.", ev)]


register(CheckSpec(key="row25", rows="25", title="Task action marked TASK UNRELATED ACTION", category=CATEGORY,
                   uses=["P2", "P5"], units=_tua_units, run=_run_row25, workers=4))


# Row 26: looking around vs turning -------------------------------------------------------------

def _idle_or_nav_units(ctx):
    def keep(c, sid):
        cl = _main_clause(c, sid)
        return _literal(c, sid) == "IDLE" or (cl is not None and cl.is_navigation)
    return _units_where(ctx, keep)


def _run_row26(ctx: CheckContext, u: Unit) -> List[Finding]:
    mo = ctx.prims.motion
    if not len(mo):
        return [_f("26", "row26", u, "unassessable", "Needs the motion timeline (P4) to split rotation from forward movement.", {})]
    s = _seg(ctx, u["segment_id"])
    m = mo[(mo.recording_id == u["recording_id"]) & (mo.track == "timeline") & (mo.t > s.start_s) & (mo.t <= s.end_s)]
    turn = float(m.camera_turn_active.mean()) if len(m) else 0.0
    fwd = float(m.forward_active.mean()) if len(m) else 0.0
    is_idle = _literal(ctx, u["segment_id"]) == "IDLE"
    ev = {"turn_share": turn, "forward_share": fwd}
    if is_idle and turn >= 0.3 and fwd < 0.1:
        return [_f("26", "row26", u, "candidate", "IDLE with sustained rotation and no walking: a turn in place or looking "
                   "around. Review only.", ev)]
    cl = _main_clause(ctx, u["segment_id"])
    if not is_idle and cl is not None and cl.verb == "turn" and turn < 0.1:
        return [_f("26", "row26", u, "candidate", "Turn caption with almost no camera rotation. Review only.", ev)]
    return [_f("26", "row26", u, "pass", "", ev)]


register(CheckSpec(key="row26", rows="26", title="Looking around confused with turning", category=CATEGORY,
                   uses=["P4"], units=_idle_or_nav_units, run=_run_row26))


# Row 27: a non-action stretch missed inside a long segment --------------------------------------

def _long_units(ctx):
    return _units_where(ctx, lambda c, sid: not _literal(c, sid)
                        and float(_seg(c, sid).get("p7_duration_pct_verb", 0) or 0) >= 0.9
                        and float(_seg(c, sid).end_s - _seg(c, sid).start_s) > 2.0)


def _run_row27(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    got = {}
    for key in ("row18", "row20"):
        f = ctx.results.get(key)
        hit = f[f.segment_id == sid] if f is not None else pd.DataFrame()
        got[key] = hit.status.iloc[0] if len(hit) else "not run"
    ev = {"row18": got["row18"], "row20": got["row20"],
          "duration_s": float(_seg(ctx, sid).end_s - _seg(ctx, sid).start_s)}
    if "candidate" in got.values():
        which = [k for k, v in got.items() if v == "candidate"]
        what = {"row18": "a pause (missed IDLE)", "row20": "sitting (missed SITTING)"}
        return [_f("27", "row27", u, "candidate", "Long segment that hides " + " and ".join(what[k] for k in which) + ".", ev)]
    if all(v in ("unassessable", "not run") for v in got.values()):
        return [_f("27", "row27", u, "unassessable", "Run rows 18 and 20 first.", ev)]
    return [_f("27", "row27", u, "pass", "No hidden pause or sitting.", ev)]


register(CheckSpec(key="row27", rows="27", title="Non-action stretch missed in a long segment", category=CATEGORY,
                   uses=["P3", "P4", "P7"], units=_long_units, run=_run_row27))


def audit_recordings(ctx: CheckContext, share: float = 0.1) -> List[str]:
    """Whole recordings picked at random for a full human audit. A duration screen misses short special
    states and recordings with no special label at all."""
    ids = sorted(ctx.prims.recordings.recording_id)
    k = max(1, int(round(len(ids) * share)))
    return sorted(random.Random(ctx.cfg.seed).sample(ids, k))


NONACTION_CHECKS = ["row18", "row19", "row20", "row21", "row22", "row23", "row24", "row25", "row26", "row27"]
