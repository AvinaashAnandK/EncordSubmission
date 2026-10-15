"""Caption errors / Hands and body attribution: sheet rows 38 to 49.

row38  The acting hand is not named.
row39  The wrong hand is named (left and right swapped).
row40  Another person's hand in view (routes to review; never captioned).
row41  An idle hand is mentioned.
row42  The other hand's action is left out.
row43  A clean transfer between hands is not recorded (caption side of row 16).
row44  A clean transfer that briefly leaves the view is not recorded.
row45  The carried object and its hand are not named during navigation.
row46  The other hand's contents are listed when they do not bear on the action (minor, report only).
row47  A body part other than a hand performs the action but is not named.
row48  A body part is named without evidence in the frames.
row49  Grip, fingers or arm path are described.

Text checks read the caption fields (P5). Frame checks use the hand timeline (P3) and object tracks
(P6) when they ran, otherwise OpenAI Decisions on sampled frames. Caption text questions go to Jev.
On a head camera the wearer's left hand appears on the left of the image; every hand question says so.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

from .. import lexicon as L
from ..clients import Question
from ..schema import Finding
from .boundary import _main_clause, boundary_units, segment_units
from .granularity import _f, _prob, _run_row16, _same_object, _seg, clip_frames, frame_at
from .runner import CheckContext, CheckSpec, Unit, register

CATEGORY = "Caption errors / Hands and body attribution"
HEAD_CAMERA = ("The camera is worn on the person's head, so their left hand appears on the left of the image and "
               "their right hand on the right.")
MOVABLE_FIXTURES = {"door", "drawer", "lid", "cabinet", "cupboard", "fridge", "refrigerator", "dishwasher", "oven",
                    "microwave", "gate"}
IDLE_HAND_RE = re.compile(r"\b(hands? (at|by) your sides?|(left|right) hand (is |stays )?(idle|empty|free|resting))\b", re.I)


def _literal(ctx, sid) -> bool:
    return ctx.prims.parsed[sid].is_fixed_literal


def _clauses(ctx, sid):
    return [c for c in ctx.prims.parsed[sid].clauses if not c.is_holding]


def _hand_action(ctx, sid) -> bool:
    c = _main_clause(ctx, sid)
    return c is not None and not _literal(ctx, sid) and not c.is_navigation


def _units(ctx, pred) -> Iterable[Unit]:
    for u in segment_units(ctx):
        if pred(ctx, u["segment_id"]):
            yield u


def _ask(ctx, q: Question, rid, start, end, n, context: str = ""):
    paths = clip_frames(ctx, rid, start, end, n)
    if not paths:
        return None
    state = f"{HEAD_CAMERA} {context}\nThe {len(paths)} images are frames from one segment, in time order."
    return ctx.clients.decisions.ask([q], state, paths)[q.name]


def _probs(ans) -> dict:
    return (ans.probs or {}) if ans is not None and ans.status == "ok" else {}


# Row 38: acting hand not named ------------------------------------------------------------------

def _run_row38(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    missing = []
    for c in _clauses(ctx, sid):
        if c.is_navigation:
            continue
        outcome_of_attempt = c.connector == "but" and c.verb in L.FAILURE_WORDS
        if not c.actor and not c.actor_shared and not outcome_of_attempt:
            missing.append(c.text)
    ev = {"clauses_without_hand": missing}
    if missing:
        return [_f("38", "row38", u, "candidate", "No hand named for: " + "; ".join(f'"{m}"' for m in missing)
                   + ". Every hand action names its hand.", ev)]
    return [_f("38", "row38", u, "pass", "Every hand action names its hand.", ev)]


register(CheckSpec(key="row38", rows="38", title="Acting hand not named", category=CATEGORY, uses=["P5"],
                   units=lambda c: _units(c, lambda x, s: not _literal(x, s)), run=_run_row38))


# Row 39: wrong hand named --------------------------------------------------------------------------

def _sided(ctx, sid) -> bool:
    c = _main_clause(ctx, sid)
    return _hand_action(ctx, sid) and c.actor in ("left hand", "right hand", "both hands") and bool(c.object)


def _run_row39(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    s, c = _seg(ctx, sid), _main_clause(ctx, sid)
    obj = c.object.split(" of ")[0]
    q = Question("which_hand", "choice", f"Which hand of the camera wearer does the action on the {obj}?",
                 {"left": "The left hand.", "right": "The right hand.", "both": "Both hands together.",
                  "unclear": "Cannot tell which hand."})
    ans = _ask(ctx, q, u["recording_id"], float(s.start_s), float(s.end_s), 5, f'The action is "{c.text}".')
    probs = _probs(ans)
    said = {"left hand": "left", "right hand": "right", "both hands": "both"}[c.actor]
    ev = {"caption_hand": said, "answer": getattr(ans, "value", None), "probs": probs}
    if not probs:
        return [_f("39", "row39", u, "unassessable", "No frame answer.", ev)]
    other = max((k for k in ("left", "right", "both") if k != said), key=lambda k: probs.get(k, 0))
    if probs.get(other, 0) >= ctx.cfg.image_flag_prob:
        return [_f("39", "row39", u, "candidate", f"The caption says {c.actor}, the frames show {other} "
                   f"(p={probs[other]:.2f}).", ev)]
    return [_f("39", "row39", u, "pass", f"Hand agrees ({ev['answer']}).", ev)]


register(CheckSpec(key="row39", rows="39", title="Wrong hand named", category=CATEGORY, uses=["P2", "P3", "P5", "P6"],
                   units=lambda c: _units(c, _sided), run=_run_row39, workers=8))


# Row 40: another person's hand ------------------------------------------------------------------------

def _run_row40(ctx: CheckContext, u: Unit) -> List[Finding]:
    s = _seg(ctx, u["segment_id"])
    ev = {}
    if len(ctx.prims.hand_dets):
        d = ctx.prims.hand_dets
        ev["unclear_owner_boxes"] = int(((d.recording_id == u["recording_id"]) & (d.t >= s.start_s) & (d.t < s.end_s)
                                         & (d.owner == "unclear")).sum())
    q = Question("other_person", "noul", "Is a hand or arm of another person, not the camera wearer, visible in any of "
                 "these frames? The camera wearer's own arms come in from the bottom of the image.")
    ans = _ask(ctx, q, u["recording_id"], float(s.start_s), float(s.end_s), 3)
    p = _prob(ans)
    ev["other_person_prob"] = p
    if p is None:
        return [_f("40", "row40", u, "unassessable", "No frame answer.", ev)]
    if p >= ctx.cfg.image_flag_prob:
        return [_f("40", "row40", u, "candidate", f"Another person's hand or arm is in view (p={p:.2f}). The reviewer checks "
                   "that the caption describes only the camera wearer's hands.", ev)]
    return [_f("40", "row40", u, "pass", "Only the camera wearer's hands.", ev)]


register(CheckSpec(key="row40", rows="40", title="Another person's hand in view", category=CATEGORY,
                   uses=["P2", "P3", "P6"], units=segment_units, run=_run_row40, workers=8))


# Row 41: idle hand mentioned ---------------------------------------------------------------------------

def _idle_hand_units(ctx):
    def keep(x, sid):
        pc = x.prims.parsed[sid]
        return not pc.is_fixed_literal and (bool(IDLE_HAND_RE.search(pc.caption)) or any(c.is_holding for c in pc.clauses))
    return _units(ctx, keep)


def _run_row41(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    s, pc = _seg(ctx, sid), ctx.prims.parsed[sid]
    m = IDLE_HAND_RE.search(pc.caption)
    if m:
        return [_f("41", "row41", u, "candidate", f'The caption mentions an idle hand ("{m.group(0)}"). Idle hands are not '
                   "mentioned.", {"phrase": m.group(0)})]
    hold = next(c for c in pc.clauses if c.is_holding)
    side = hold.held_actor.replace(" hand", "").replace(" hands", "")
    obj = hold.held_object.split(" of ")[0]
    who = "both hands hold" if side == "both" else f"the camera wearer's {side} hand holds"
    q = Question("holds", "noul", f"In these frames, {who} the {obj}?")
    ans = _ask(ctx, q, u["recording_id"], float(s.start_s), float(s.end_s), 3)
    p = _prob(ans)
    ev = {"held_object": obj, "held_hand": hold.held_actor, "holds_prob": p}
    if p is None:
        return [_f("41", "row41", u, "unassessable", "No frame answer.", ev)]
    if p <= 1 - ctx.cfg.image_flag_prob:
        return [_f("41", "row41", u, "candidate", f"The caption says the {hold.held_actor} holds the {obj}, but it does not "
                   f"seem to (p={p:.2f}). An idle hand is not mentioned.", ev)]
    return [_f("41", "row41", u, "pass", f"The {hold.held_actor} holds the {obj}.", ev)]


register(CheckSpec(key="row41", rows="41", title="Idle hand mentioned", category=CATEGORY, uses=["P2", "P3", "P5"],
                   units=_idle_hand_units, run=_run_row41, workers=8))


# Row 42: other hand's action left out --------------------------------------------------------------------

def _one_hand(ctx, sid) -> bool:
    if not _hand_action(ctx, sid):
        return False
    pc = ctx.prims.parsed[sid]
    actors = {c.actor for c in pc.clauses if c.actor}
    return actors in ({"left hand"}, {"right hand"}) and not any(c.is_holding for c in pc.clauses)


def _run_row42(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    s, c = _seg(ctx, sid), _main_clause(ctx, sid)
    other = "left" if c.actor == "right hand" else "right"
    # Holding something still is not an action of its own. Whether held contents must be named is
    # rows 45 and 46, so it is left out here.
    q = Question("other_hand_acts", "noul",
                 f"Besides the action in the caption, does the camera wearer's {other} hand do an action of its own: "
                 "pick an object up, put one down, or move or work on a different object? Holding an object still, "
                 "resting, or settling onto an object the other hand already holds does not count.")
    ans = _ask(ctx, q, u["recording_id"], float(s.start_s), float(s.end_s), 5, f'The caption is "{s.caption}".')
    p = _prob(ans)
    ev = {"named_hand": c.actor, "other_hand": other, "other_hand_acts_prob": p}
    if p is None:
        return [_f("42", "row42", u, "unassessable", "No frame answer.", ev)]
    if p >= ctx.cfg.image_flag_prob:
        return [_f("42", "row42", u, "candidate", f"The {other} hand also acts (p={p:.2f}) but the caption names only the "
                   f"{c.actor}.", ev)]
    return [_f("42", "row42", u, "pass", f"The {other} hand does not act.", ev)]


register(CheckSpec(key="row42", rows="42", title="Other hand's action left out", category=CATEGORY,
                   uses=["P2", "P3", "P5"], units=lambda c: _units(c, _one_hand), run=_run_row42, workers=8))


# Rows 43 and 44: transfers -------------------------------------------------------------------------------

def _run_row43(ctx: CheckContext, u: Unit) -> List[Finding]:
    # The same comparison as row 16 (cached), read from the caption side.
    out = []
    for f in _run_row16(ctx, u):
        msg = f.message.replace("Check the hands (row 39).", "Missing transfer or wrong hand: the reviewer decides.")
        out.append(Finding(row="43", check="row43", recording_id=f.recording_id, segment_id=f.segment_id,
                           status=f.status, message=msg, evidence=f.evidence))
    return out


register(CheckSpec(key="row43", rows="43", title="Transfer between hands not recorded", category=CATEGORY,
                   uses=["P2", "P5"], units=boundary_units, run=_run_row43, workers=8, unit="boundary"))


def _run_row44(ctx: CheckContext, u: Unit) -> List[Finding]:
    import json

    prev = ctx.results.get("row43")
    hit = prev[prev.segment_id == u["segment_id"]] if prev is not None else None
    base = {"prev_id": u["prev_id"], "t": u["t"]}
    if hit is None or hit.empty:
        return [_f("44", "row44", u, "unassessable", "Run row 43 first.", base)]
    ev = {**json.loads(hit.evidence.iloc[0]), **base}
    if hit.status.iloc[0] not in ("candidate", "report"):
        return [_f("44", "row44", u, "pass", "No hand change on the same object here.", ev)]
    paths = clip_frames(ctx, u["recording_id"], u["t"] - 0.6, u["t"] + 0.6, 5)
    q = Question("transfer_seen", "noul", "Is the moment the object passes from one hand to the other visible in these "
                 "frames, with both hands on it?")
    ans = ctx.clients.decisions.ask([q], f"{HEAD_CAMERA} Frames around a cut, in time order.", paths)[q.name] if paths else None
    p = _prob(ans)
    ev["transfer_seen_prob"] = p
    if p is None:
        return [_f("44", "row44", u, "unassessable", "No frame answer.", ev)]
    if p <= 1 - ctx.cfg.image_flag_prob and hit.status.iloc[0] == "candidate":
        return [_f("44", "row44", u, "candidate", "The object changes hands out of view. If it is unmistakable from the "
                   "frames either side, the transfer must still be described.", ev)]
    return [_f("44", "row44", u, "pass", "The transfer is visible, or there is none.", ev)]


register(CheckSpec(key="row44", rows="44", title="Transfer out of view not recorded", category=CATEGORY,
                   uses=["P2", "P3"], units=boundary_units, run=_run_row44, workers=8, unit="boundary"))


# Row 45: carried object not named during navigation ---------------------------------------------------------

def _nav(ctx, sid) -> bool:
    c = _main_clause(ctx, sid)
    return c is not None and c.is_navigation


def _run_row45(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    s, pc = _seg(ctx, sid), ctx.prims.parsed[sid]
    q = Question("carrying", "choice", "Does the camera wearer carry an object while moving? In which hand?",
                 {"none": "Nothing is carried.", "left": "Carried in the left hand.", "right": "Carried in the right hand.",
                  "both": "Carried with both hands.", "unclear": "Cannot tell."})
    ans = _ask(ctx, q, u["recording_id"], float(s.start_s), float(s.end_s), 4)
    probs = _probs(ans)
    named = [c.held_actor for c in pc.clauses if c.is_holding]
    ev = {"named_carry": named, "answer": getattr(ans, "value", None)}
    if not probs:
        return [_f("45", "row45", u, "unassessable", "No frame answer.", ev)]
    carry = max(("left", "right", "both"), key=lambda k: probs.get(k, 0))
    if probs[carry] >= ctx.cfg.image_flag_prob and not named:
        return [_f("45", "row45", u, "candidate", f"Something is carried ({carry}, p={probs[carry]:.2f}) but the caption "
                   "names no carried object. Omitting it implies empty hands.", ev)]
    return [_f("45", "row45", u, "pass", "Carried objects agree with the caption.", ev)]


register(CheckSpec(key="row45", rows="45", title="Carried object not named during navigation", category=CATEGORY,
                   uses=["P2", "P3", "P6"], units=lambda c: _units(c, _nav), run=_run_row45, workers=8))


# Row 46: other hand's contents listed when they do not bear on the action -----------------------------------

def _held_with_action(ctx, sid) -> bool:
    pc = ctx.prims.parsed[sid]
    return _hand_action(ctx, sid) and any(c.is_holding for c in pc.clauses)


def _run_row46(ctx: CheckContext, u: Unit) -> List[Finding]:
    s = _seg(ctx, u["segment_id"])
    q = Question("bears_on", "noul", "In this caption, does the object held in the other hand matter to the main action: "
                 "is it used in it, supported by it, or the thing being worked on?")
    ans = ctx.clients.jev.ask([q], {"caption": str(s.caption)})[q.name]
    p = _prob(ans)
    ev = {"bears_on_prob": p}
    if p is None:
        return [_f("46", "row46", u, "unassessable", "No text answer.", ev)]
    if p < 0.3:
        return [_f("46", "row46", u, "report", f"The other hand's contents may not bear on the action (p={p:.2f}). Minor: "
                   "sampled for review, not scored. The guideline's own fork example lists such contents.", ev)]
    return [_f("46", "row46", u, "pass", "The held object bears on the action.", ev)]


register(CheckSpec(key="row46", rows="46", title="Other hand's contents listed needlessly", category=CATEGORY,
                   uses=["P5"], units=lambda c: _units(c, _held_with_action), run=_run_row46))


# Rows 47 and 48: body parts other than hands -----------------------------------------------------------------

def _fixture_action(ctx, sid) -> bool:
    c = _main_clause(ctx, sid)
    return _hand_action(ctx, sid) and (c.object_head or "") in MOVABLE_FIXTURES


def _run_row47(ctx: CheckContext, u: Unit) -> List[Finding]:
    s, c = _seg(ctx, u["segment_id"]), _main_clause(ctx, u["segment_id"])
    q = Question("mover", "choice", f"What moves the {c.object_head}?",
                 {"hand": "A hand.", "hip": "The hip.", "foot": "A foot.", "knee": "A knee.", "elbow": "An elbow.",
                  "unknown": "Cannot tell, or something else such as another person or gravity."})
    ans = _ask(ctx, q, u["recording_id"], float(s.start_s), float(s.end_s), 4)
    probs = _probs(ans)
    ev = {"caption_actor": c.actor, "answer": getattr(ans, "value", None), "probs": probs}
    if not probs:
        return [_f("47", "row47", u, "unassessable", "No frame answer.", ev)]
    part = max(("hip", "foot", "knee", "elbow"), key=lambda k: probs.get(k, 0))
    if probs[part] >= ctx.cfg.image_flag_prob and c.actor not in L.BODY_PARTS:
        return [_f("47", "row47", u, "candidate", f"The {c.object_head} is moved by the {part} (p={probs[part]:.2f}), but the "
                   f"caption names {c.actor or 'no body part'}.", ev)]
    return [_f("47", "row47", u, "pass", f"Mover: {ev['answer']}.", ev)]


register(CheckSpec(key="row47", rows="47", title="Non-hand body part not named", category=CATEGORY,
                   uses=["P2", "P3", "P6"], units=lambda c: _units(c, _fixture_action), run=_run_row47, workers=8))


def _body_part_caption(ctx, sid) -> bool:
    return not _literal(ctx, sid) and bool(L.BODY_RE.search(ctx.prims.parsed[sid].caption))


def _run_row48(ctx: CheckContext, u: Unit) -> List[Finding]:
    s, c = _seg(ctx, u["segment_id"]), _main_clause(ctx, u["segment_id"])
    part = L.BODY_RE.search(s.caption).group("part")
    obj = (c.object_head if c else "") or "object"
    q = Question("hand_touches", "noul", f"Does a hand of the camera wearer touch the {obj} while it moves?")
    ans = _ask(ctx, q, u["recording_id"], float(s.start_s) - 1.0, float(s.end_s) + 1.0, 5)
    p = _prob(ans)
    ev = {"body_part": part, "hand_touches_prob": p}
    if p is not None and p >= ctx.cfg.image_flag_prob:
        return [_f("48", "row48", u, "candidate", f"Caption names the {part}, but a hand touches the {obj} while it moves "
                   f"(p={p:.2f}).", ev)]
    return [_f("48", "row48", u, "candidate", f"Caption names the {part}. Every body part claim goes to review: the "
               "frames either side must make it unmistakable.", ev)]


register(CheckSpec(key="row48", rows="48", title="Body part named without evidence", category=CATEGORY,
                   uses=["P2", "P3", "P6"], units=lambda c: _units(c, _body_part_caption), run=_run_row48, workers=8))


# Row 49: grip described ----------------------------------------------------------------------------------------

def _grip_caption(ctx, sid) -> bool:
    return not _literal(ctx, sid) and bool(L.GRIP_RE.search(ctx.prims.parsed[sid].caption))


def _run_row49(ctx: CheckContext, u: Unit) -> List[Finding]:
    s = _seg(ctx, u["segment_id"])
    word = L.GRIP_RE.search(s.caption).group(1)
    q = Question("describes_grip", "noul", f'Does the word "{word}" in this caption describe how the hand holds or moves '
                 "(grip, fingers, arm path), rather than naming an object or the action itself?")
    ans = ctx.clients.jev.ask([q], {"caption": str(s.caption)})[q.name]
    p = _prob(ans)
    ev = {"word": word, "describes_grip_prob": p}
    if p is None:
        return [_f("49", "row49", u, "unassessable", "No text answer.", ev)]
    if p >= 0.5:
        return [_f("49", "row49", u, "candidate", f'Describes the grip ("{word}"). Warning: the fix deletes the phrase in '
                   "a proposed edit; the original is kept.", ev)]
    return [_f("49", "row49", u, "pass", f'"{word}" names an object or the action.', ev)]


register(CheckSpec(key="row49", rows="49", title="Grip or arm path described", category=CATEGORY, uses=["P5"],
                   units=lambda c: _units(c, _grip_caption), run=_run_row49))


ATTRIBUTION_CHECKS = ["row38", "row39", "row40", "row41", "row42", "row43", "row44", "row45", "row46", "row47",
                      "row48", "row49"]
