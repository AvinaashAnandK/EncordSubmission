"""Caption errors / Completeness of the instruction: sheet rows 35 to 37.

row35  A required element is missing: verb, object, actor (hand or body part), landmark, or a visible place.
row36  An action with a defined end point does not state it.
row37  An action that briefly leaves the view but is unmistakable is not described.

The field checks read the caption fields (P5) and need no model. A missing place, a reached end state and
an object that changes hands out of view are read from frames: P3 and P6 when they ran, otherwise OpenAI
Decisions on one or two frames. Caption text questions go to Jev.
"""
from __future__ import annotations

from typing import Iterable, List

from .. import lexicon as L
from ..clients import Question
from ..schema import Finding
from .boundary import _main_clause, segment_units
from .faithfulness import DESTINATION_VERBS
from .granularity import _f, _prob, _seg, frame_at
from .runner import CheckContext, CheckSpec, Unit, register

CATEGORY = "Caption errors / Completeness of the instruction"
# Verbs that end with the object in a new place. "Push the parts together" or a failed attempt does not.
MOVE_TO_VERBS = DESTINATION_VERBS - {"push", "empty", "pour", "spread out"}


def _literal(ctx, sid) -> bool:
    return ctx.prims.parsed[sid].is_fixed_literal


def _units(ctx, pred) -> Iterable[Unit]:
    for u in segment_units(ctx):
        if pred(ctx, u["segment_id"]):
            yield u


# Row 35: a required element missing ------------------------------------------------------------

def _missing_fields(cl) -> List[str]:
    if cl.is_holding:
        return []
    if cl.is_navigation:
        return [] if (cl.landmarks or cl.facing) else ["landmark"]
    miss = []
    if not cl.verb:
        miss.append("verb")
    if not cl.object and cl.verb not in L.FAILURE_WORDS:
        miss.append("object")
    # "... but fail to join them" is the outcome of the same attempt, so it shares that attempt's hand.
    outcome_of_attempt = cl.connector == "but" and cl.verb in L.FAILURE_WORDS
    if not cl.actor and not cl.actor_shared and not outcome_of_attempt:
        miss.append("hand or body part")
    return miss


def _run_row35(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    s = _seg(ctx, sid)
    clauses = ctx.prims.parsed[sid].clauses
    gaps = {i: m for i, cl in enumerate(clauses) if (m := _missing_fields(cl))}
    ev = {"missing": {clauses[i].text: m for i, m in gaps.items()}}
    # A clause that moves an object somewhere but names no place: ask whether a place is visible.
    no_place = next((cl for cl in clauses if not cl.is_holding and not cl.is_navigation and not cl.place
                     and not cl.intent and (cl.verb in MOVE_TO_VERBS or cl.verb_key in MOVE_TO_VERBS)), None)
    place_seen = None
    if no_place is not None:
        obj = (no_place.object or "the object").split(" of ")[0]
        last = frame_at(ctx, u["recording_id"], float(s.end_s) - 0.1)
        q = Question("place_visible", "noul",
                     f"Is the {obj} now resting on or in a place you can see and name, such as a table, a plate, a "
                     "shelf or a container?")
        ans = ctx.clients.decisions.ask([q], "The last frame of one segment of a first person video.", [last])[q.name] if last else None
        place_seen = _prob(ans)
        ev.update({"no_place_clause": no_place.text, "place_visible_prob": place_seen,
                   "direction": no_place.direction})
    msgs = [f'"{clauses[i].text}" has no {", ".join(m)}' for i, m in gaps.items()]
    if place_seen is not None and place_seen >= ctx.cfg.image_flag_prob:
        d = f' (it gives only a direction, "{no_place.direction}")' if no_place.direction else ""
        msgs.append(f'"{no_place.text}" names no place, but a place is visible at the end (p={place_seen:.2f}){d}')
    if msgs:
        return [_f("35", "row35", u, "candidate", "Missing: " + "; ".join(msgs) + ".", ev)]
    return [_f("35", "row35", u, "pass", "Every required element is present.", ev)]


register(CheckSpec(key="row35", rows="35", title="Required element missing", category=CATEGORY,
                   uses=["P2", "P5", "P6"], units=lambda c: _units(c, lambda x, s: not _literal(x, s)),
                   run=_run_row35, workers=8))


# Row 36: a defined end point not stated ------------------------------------------------------------

def _end_state_clause(ctx, sid):
    return next((cl for cl in ctx.prims.parsed[sid].clauses
                 if cl.verb_key in L.END_STATE_VERBS or cl.verb in L.END_STATE_VERBS), None)


def _run_row36(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    s, cl = _seg(ctx, sid), _end_state_clause(ctx, sid)
    q = Question("states_end", "noul",
                 "Does this caption state where the action stops, its end point? A verb such as close or turn off "
                 'already states it. "Until it is closed" or "until full" states it. A bare "twist the cap" does not.')
    ans = ctx.clients.jev.ask([q], {"caption": str(s.caption)})[q.name]
    p = _prob(ans)
    ev = {"verb": cl.verb_key, "states_end_prob": p}
    if p is None:
        return [_f("36", "row36", u, "unassessable", "No text answer.", ev)]
    if p >= 0.5:
        return [_f("36", "row36", u, "pass", f"The end point is stated (p={p:.2f}).", ev)]
    obj = (cl.object or "the object").split(" of ")[0]
    last = frame_at(ctx, u["recording_id"], float(s.end_s) - 0.1)
    q2 = Question("end_reached", "noul", f"Has the {obj} reached a clear end state in this frame, such as fully "
                  "closed, fully open, full, empty or folded?")
    a2 = ctx.clients.decisions.ask([q2], "The last frame of one segment of a first person video.", [last])[q2.name] if last else None
    p2 = _prob(a2)
    ev["end_reached_prob"] = p2
    if p2 is not None and p2 >= ctx.cfg.image_flag_prob:
        return [_f("36", "row36", u, "candidate", f"The {cl.verb_key} reaches an end state in the frames (p={p2:.2f}) "
                   "but the caption does not state it.", ev)]
    return [_f("36", "row36", u, "pass", "No end point shown, so none should be stated.", ev)]


register(CheckSpec(key="row36", rows="36", title="Defined end point not stated", category=CATEGORY,
                   uses=["P2", "P5"], units=lambda c: _units(c, lambda x, s: not _literal(x, s)
                                                             and _end_state_clause(x, s) is not None),
                   run=_run_row36, workers=8))


# Row 37: an action out of view not described --------------------------------------------------------

def _row37_units(ctx):
    seg = ctx.prims.segments.sort_values(["recording_id", "segment_index"])
    for rid, g in seg.groupby("recording_id"):
        ids = list(g.segment_id)
        for i, sid in enumerate(ids):
            s = _seg(ctx, sid)
            lit = ctx.prims.parsed[sid].literal if ctx.prims.parsed[sid].is_fixed_literal else ""
            short = float(s.end_s - s.start_s) < 1.0
            next_to_hand = any(not ctx.prims.parsed[n].is_fixed_literal
                               for n in ids[max(0, i - 1):i] + ids[i + 1:i + 2])
            if lit == "IDLE" or (short and next_to_hand):
                yield {"recording_id": rid, "segment_id": sid}


def _run_row37(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    s = _seg(ctx, sid)
    a = frame_at(ctx, u["recording_id"], float(s.start_s) + 0.05)
    b = frame_at(ctx, u["recording_id"], float(s.end_s) - 0.05)
    q = Question("held_change", "choice",
                 "Compare what each hand of the camera wearer holds in frame 1 and in frame 2.",
                 {"moved_to_other_hand": "An object held in one hand in frame 1 is in the other hand in frame 2.",
                  "left_a_hand": "An object held in frame 1 is no longer held in frame 2.",
                  "picked_up": "A hand holds an object in frame 2 that it did not hold in frame 1.",
                  "no_change": "Each hand holds the same thing, or nothing, in both frames.",
                  "unclear": "Cannot tell."})
    ans = ctx.clients.decisions.ask([q], "Frame 1 is the start and frame 2 is the end of one segment.", [a, b])[q.name] if a and b else None
    probs = (ans.probs or {}) if ans is not None and ans.status == "ok" else {}
    change = max(("moved_to_other_hand", "left_a_hand", "picked_up"), key=lambda k: probs.get(k, 0)) if probs else None
    ev = {"answer": getattr(ans, "value", None), "change": change, "change_prob": probs.get(change) if change else None}
    if not probs:
        return [_f("37", "row37", u, "unassessable", "No frame answer.", ev)]
    is_idle = ctx.prims.parsed[sid].is_fixed_literal
    if probs[change] >= ctx.cfg.image_flag_prob and is_idle:
        what = {"moved_to_other_hand": "moves to the other hand", "left_a_hand": "leaves a hand",
                "picked_up": "appears in a hand"}[change]
        return [_f("37", "row37", u, "candidate", f"Marked IDLE but an object {what} during the segment "
                   f"(p={probs[change]:.2f}). If the action is unmistakable from the frames either side, describe it.", ev)]
    return [_f("37", "row37", u, "pass", f"No undescribed change in what the hands hold ({ev['answer']}).", ev)]


register(CheckSpec(key="row37", rows="37", title="Action out of view not described", category=CATEGORY,
                   uses=["P2", "P3", "P6"], units=_row37_units, run=_run_row37, workers=8))


COMPLETENESS_CHECKS = ["row35", "row36", "row37"]
