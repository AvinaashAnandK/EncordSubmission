"""Caption errors / Faithfulness to what happened: sheet rows 28 to 34.

row28  The caption describes what was meant, not what happened (rides on rows 30 and 33).
row29  A failure, aborted attempt or accident is softened or left out.
row30  Intent wording ("try to", "attempt to") with no stated outcome.
row31  The caption names an object that is not visible in the segment.
row32  The caption names a destination that is not visible, or the object does not end up there.
row33  The verb is wrong or the described action did not happen (three tiers).
row34  A confident guess where IDLE or a fallback noun was needed.

Object and place questions use SAM 3 (P6) when it ran. Without it, OpenAI Decisions answers closed
questions on sampled frames. Caption text questions go to Jev. Sol is used only for tier 3 of row 33.
A missed detection never proves that an object is out of view, so "not visible" is a candidate for
review, not a verdict.
"""
from __future__ import annotations

import random
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .. import lexicon as L
from ..clients import Question
from ..schema import Finding
from .boundary import _main_clause, segment_units
from .granularity import _f, _prob, _seg, clip_frames, frame_at
from .runner import CheckContext, CheckSpec, Unit, register

CATEGORY = "Caption errors / Faithfulness to what happened"


def _literal(ctx, sid) -> bool:
    return ctx.prims.parsed[sid].is_fixed_literal


def _actions(ctx, sid):
    return [c for c in ctx.prims.parsed[sid].clauses if not c.is_holding]


def _hand_action(ctx, sid) -> bool:
    c = _main_clause(ctx, sid)
    return c is not None and not _literal(ctx, sid) and not c.is_navigation


def _has_failure(ctx, sid) -> bool:
    pc = ctx.prims.parsed[sid]
    return any(c.has_failure_word for c in pc.clauses)


def _objects(ctx, sid) -> List[str]:
    """Object phrases a viewer should be able to see, without pronouns or body parts."""
    out = []
    for c in ctx.prims.parsed[sid].clauses:
        for p in (c.object, c.held_object):
            p = (p or "").strip()
            if p and p not in ("it", "them", "hands", "hand") and p not in out:
                out.append(p)
    return out


def _units(ctx, pred) -> Iterable[Unit]:
    for u in segment_units(ctx):
        if pred(ctx, u["segment_id"]):
            yield u


def _decide(ctx, questions, rid, paths, context) -> dict:
    if not paths:
        return {}
    return ctx.clients.decisions.ask(questions, context, paths)


def _review_sample(ctx, key, sid, share) -> bool:
    return random.Random(f"{ctx.cfg.seed}:{key}:{sid}").random() < share


# Row 30: intent wording with no outcome -----------------------------------------------------

def _run_row30(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    cap = str(_seg(ctx, sid).caption)
    m = L.INTENT_RE.search(cap)
    if not m:
        return [_f("30", "row30", u, "pass", "No intent wording.", {})]
    q = Question("states_outcome", "noul",
                 "Does this caption state what actually happened, including whether the attempt failed or succeeded? "
                 'An outcome can follow "but" or "then", or be the main verb, e.g., "Fail to open the jar".')
    ans = ctx.clients.jev.ask([q], {"caption": cap})[q.name]
    p = _prob(ans)
    ev = {"intent": m.group(1), "states_outcome_prob": p, "source": getattr(ans, "source", "")}
    if p is None:
        return [_f("30", "row30", u, "unassessable", "No text answer.", ev)]
    if p < 0.5:
        return [_f("30", "row30", u, "candidate", f'Intent wording "{m.group(1)}" with no stated outcome (p={p:.2f}).', ev)]
    return [_f("30", "row30", u, "pass", f'Intent wording "{m.group(1)}" with its outcome stated.', ev)]


register(CheckSpec(key="row30", rows="30", title="Intent wording with no outcome", category=CATEGORY,
                   uses=["P5"], units=lambda c: _units(c, lambda x, s: not _literal(x, s)), run=_run_row30))


# Row 29: failure softened or left out ---------------------------------------------------------

def _run_row29(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    s = _seg(ctx, sid)
    if _has_failure(ctx, sid) or L.INTENT_RE.search(str(s.caption)):
        return [_f("29", "row29", u, "pass", "The caption already records a failure or attempt.", {})]
    q = Question("something_went_wrong", "noul",
                 "Does anything go wrong in this clip: a miss, a slip, a drop, a spill, or a reach pulled back before "
                 "the hand takes the object?")
    paths = clip_frames(ctx, u["recording_id"], float(s.start_s), float(s.end_s), 6)
    ans = _decide(ctx, [q], u["recording_id"], paths, f'The caption of this clip is "{s.caption}". The images are '
                  f"{len(paths)} frames in time order.").get(q.name)
    p = _prob(ans)
    unedited = str(s.get("accepted_type", "")) == "accepted_without_edits"
    ev = {"went_wrong_prob": p, "suggestion_accepted_unedited": unedited, "source": getattr(ans, "source", "")}
    if p is None:
        return [_f("29", "row29", u, "unassessable", "No frame answer.", ev)]
    if p >= ctx.cfg.image_flag_prob:
        extra = " The caption was a suggestion accepted without edits, and suggestions are weakest on failures." if unedited else ""
        return [_f("29", "row29", u, "candidate", f"Something goes wrong in the clip but the caption records no failure "
                   f"(p={p:.2f}).{extra}", ev)]
    return [_f("29", "row29", u, "pass", "Nothing goes wrong in the sampled frames.", ev)]


register(CheckSpec(key="row29", rows="29", title="Failure softened or left out", category=CATEGORY,
                   uses=["P2", "P3", "P5", "P6", "P8"], units=lambda c: _units(c, _hand_action), run=_run_row29, workers=8))


# Row 31: object not visible ---------------------------------------------------------------------

def _run_row31(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    objs = _objects(ctx, sid)
    if not objs:
        return [_f("31", "row31", u, "pass", "No visible object named.", {})]
    s = _seg(ctx, sid)
    ev = {"objects": objs}
    if len(ctx.prims.objects):
        share = s.get("p7_object_found_share")
        ev["sam3_found_share"] = None if pd.isna(share) else float(share)
    paths = clip_frames(ctx, u["recording_id"], float(s.start_s), float(s.end_s), 5)
    qs = [Question(f"visible_{i}", "noul", f'Is "{o}" visible in at least one of these frames?') for i, o in enumerate(objs)]
    ans = _decide(ctx, qs, u["recording_id"], paths, f"{len(paths)} frames from one segment of a first person video.")
    probs = {o: _prob(ans.get(q.name)) for o, q in zip(objs, qs)}
    ev["visible_prob"] = probs
    if all(v is None for v in probs.values()):
        return [_f("31", "row31", u, "unassessable", "No frame answer.", ev)]
    gone = [o for o, v in probs.items() if v is not None and v <= 1 - ctx.cfg.image_flag_prob]
    if gone:
        return [_f("31", "row31", u, "candidate", "Named but not seen in the sampled frames: " + ", ".join(
            f"{o} (p visible={probs[o]:.2f})" for o in gone) + ". A miss does not prove it is out of view.", ev)]
    return [_f("31", "row31", u, "pass", "Every named object is visible.", ev)]


register(CheckSpec(key="row31", rows="31", title="Object named but not visible", category=CATEGORY,
                   uses=["P2", "P5", "P6", "P7"], units=lambda c: _units(c, lambda x, s: not _literal(x, s)),
                   run=_run_row31, workers=8))


# Row 32: destination not visible ------------------------------------------------------------------

# Verbs whose place is a destination. "Rinse the fork under the tap" names where the action happens,
# not where the fork ends up, so it is not checked here.
DESTINATION_VERBS = {"put", "place", "insert", "drop", "set", "move", "transfer", "hang", "throw", "empty",
                     "pour", "lay", "push", "slide", "return", "stack", "load", "spread out"}


def _dest_clause(ctx, sid):
    return next((c for c in _actions(ctx, sid) if c.place and c.verb_key in DESTINATION_VERBS | {"put down"}
                 or (c.place and c.verb in DESTINATION_VERBS)), None)


def _with_place(ctx, sid) -> bool:
    return _hand_action(ctx, sid) and _dest_clause(ctx, sid) is not None


def _run_row32(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    s = _seg(ctx, sid)
    cl = _dest_clause(ctx, sid)
    obj = (cl.object or "the object").split(" of ")[0]
    end = float(s.end_s)
    paths = clip_frames(ctx, u["recording_id"], max(float(s.start_s), end - 1.0), end, 3)
    qs = [Question("place_visible", "noul", f'Is "{cl.place}" visible in these frames?'),
          Question("at_place", "noul", f'In the last frame, is the {obj} {cl.place_prep} the {cl.place}?')]
    ans = _decide(ctx, qs, u["recording_id"], paths, "The last second of one segment of a first person video, in time order.")
    pv, pa = _prob(ans.get("place_visible")), _prob(ans.get("at_place"))
    ev = {"place": cl.place, "prep": cl.place_prep, "object": obj, "place_visible_prob": pv, "at_place_prob": pa}
    if pv is None and pa is None:
        return [_f("32", "row32", u, "unassessable", "No frame answer.", ev)]
    low = 1 - ctx.cfg.image_flag_prob
    if pv is not None and pv <= low:
        return [_f("32", "row32", u, "candidate", f'The destination "{cl.place}" is not seen at the end of the segment '
                   f"(p={pv:.2f}). If it is out of view, the caption should stop at what is visible.", ev)]
    if pa is not None and pa <= low:
        return [_f("32", "row32", u, "candidate", f'The {obj} does not end up {cl.place_prep} the {cl.place} (p={pa:.2f}).', ev)]
    return [_f("32", "row32", u, "pass", "The destination is visible and the object ends up there.", ev)]


register(CheckSpec(key="row32", rows="32", title="Destination not visible", category=CATEGORY,
                   uses=["P2", "P5", "P6"], units=lambda c: _units(c, _with_place), run=_run_row32, workers=8))


# Row 33: wrong verb, or the action did not happen ---------------------------------------------------

TIER3_OPTIONS = {
    "correct": "The caption describes the clip: the right action, object and hand.",
    "wrong_action": "The action in the clip is different from the caption.",
    "wrong_object": "The object in the clip is different from the caption.",
    "wrong_hand": "A different hand or body part does the action.",
    "did_not_happen": "The action starts but its described result never happens, e.g., the object is never picked up or never reaches the place.",
    "extra_actions": "The caption is right, but the clip also shows another action.",
}
TIER3_SHORT = {"wrong_action": "wrong action", "wrong_object": "wrong object", "wrong_hand": "wrong hand",
               "did_not_happen": "the described result never happens", "extra_actions": "the clip also shows another action",
               "": "not correct"}


def _run_row33(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    s = _seg(ctx, sid)
    rid, start, end = u["recording_id"], float(s.start_s), float(s.end_s)
    ev = {}
    # Tier 1 (P3): the verb against the contact pattern.
    c = _main_clause(ctx, sid)
    t1 = None
    if len(ctx.prims.hand_timeline) and c is not None:
        ht = ctx.prims.hand_timeline
        h = ht[(ht.recording_id == rid) & (ht.t >= start) & (ht.t < end) & ht.state.isin(["contact", "no_contact"])].sort_values("t")
        if len(h) >= 4:
            first, last = h.state.iloc[: len(h) // 4], h.state.iloc[-(len(h) // 4):]
            starts = (first == "no_contact").mean() > 0.5 and (last == "contact").mean() > 0.5
            ends = (first == "contact").mean() > 0.5 and (last == "no_contact").mean() > 0.5
            if c.verb_key in ("pick up", "grab", "take", "lift"):
                t1 = starts
            elif c.verb in ("place", "put", "drop", "set"):
                t1 = ends
        ev["tier1_matches"] = t1
    # Tier 2: the first and last frames against the start and end of the caption.
    a, b = frame_at(ctx, rid, start + 0.1), frame_at(ctx, rid, end - 0.1)
    q2 = Question("start_end_match", "noul",
                  f'Frame 1 is the start and frame 2 is the end of a segment captioned "{s.caption}". '
                  "Do the two frames match the start and the end of what this caption says happens?")
    a2 = _decide(ctx, [q2], rid, [x for x in (a, b) if x], "Two frames from a first person video.").get(q2.name)
    p2 = _prob(a2)
    ev["tier2_prob"] = p2
    marked = (t1 is False) or (p2 is not None and p2 < 0.5)
    audit = random.Random(f"{ctx.cfg.seed}:row33:{sid}").random() < 0.05
    ev["random_audit"] = audit
    if not marked and not audit:
        if p2 is None and t1 is None:
            return [_f("33", "row33", u, "unassessable", "No tier 1 or tier 2 answer.", ev)]
        return [_f("33", "row33", u, "pass", f"Start and end frames match the caption (p={p2:.2f}).", ev)]
    # Tier 3: Sol on the clip, with the kind of error.
    q3 = Question("caption_check", "choice", "Does this caption describe what happens in the clip? If not, what is wrong?",
                  TIER3_OPTIONS)
    paths = clip_frames(ctx, rid, start, end, 8)
    a3 = ctx.clients.sol.ask(q3, f'Caption: "{s.caption}". The images are {len(paths)} frames of the clip in time order.',
                             paths) if paths else None
    probs = (a3.probs or {}) if a3 is not None and a3.status == "ok" else {}
    pc = probs.get("correct")
    reason = max((k for k in probs if k != "correct"), key=lambda k: probs[k], default="")
    ev.update({"tier3_correct_prob": pc, "tier3_reason": reason, "tier3_reason_prob": probs.get(reason),
               "tier3_source": getattr(a3, "source", "")})
    if pc is None:
        return [_f("33", "row33", u, "unassessable", "Marked by tier 1 or 2, but no clip answer.", ev)]
    if pc < ctx.cfg.image_flag_prob and reason == "extra_actions":
        # The caption is right; the extra action is a boundary or merge question (rows 2, 3 and 8).
        return [_f("33", "row33", u, "report", f"Caption right, but the clip also shows another action "
                   f"(p={probs[reason]:.2f}). See rows 2, 3 and 8.", ev)]
    if pc < ctx.cfg.image_flag_prob:
        return [_f("33", "row33", u, "candidate", f"Sol: {TIER3_SHORT[reason]} (p={probs[reason]:.2f}; p correct={pc:.2f}). "
                   f"Tier 2 p={p2 if p2 is None else round(p2, 2)}.", ev)]
    why = "random audit" if audit and not marked else "tier 2 mark cleared"
    return [_f("33", "row33", u, "pass", f"Sol says the caption is correct (p={pc:.2f}; {why}).", ev)]


register(CheckSpec(key="row33", rows="33", title="Wrong verb or action did not happen", category=CATEGORY,
                   uses=["P2", "P3", "P5"], units=lambda c: _units(c, lambda x, s: not _literal(x, s)),
                   run=_run_row33, workers=8))


# Row 34: confident guess -----------------------------------------------------------------------------

def _specific_object(ctx, sid) -> bool:
    c = _main_clause(ctx, sid)
    return (_hand_action(ctx, sid) and c is not None and bool(c.object_head)
            and c.object_head not in L.FALLBACK_NOUNS and c.object not in ("it", "them", "hands"))


def _run_row34(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    s, c = _seg(ctx, sid), _main_clause(ctx, sid)
    fr = ctx.prims.frames
    f = fr[(fr.recording_id == u["recording_id"]) & fr.is_timeline & (fr.t >= s.start_s) & (fr.t <= s.end_s)]
    if f.empty:
        return [_f("34", "row34", u, "unassessable", "No frames.", {})]
    best = f.loc[f.blur.idxmax()]
    obj = c.object.split(" of ")[0]
    q = Question("identifiable", "choice",
                 f'The caption calls the object "{obj}". In this image, can a person tell that the object is a {c.object_head}?',
                 {"identifiable": f"The object is clearly a {c.object_head}.",
                  "not_identifiable": "The object is too small, blurred or hidden to tell what it is.",
                  "different_object": f"The object is clearly something other than a {c.object_head}.",
                  "unclear": "Cannot tell which object is meant."})
    ans = _decide(ctx, [q], u["recording_id"], [best.path], "The sharpest frame of one segment.").get(q.name)
    probs = (ans.probs or {}) if ans is not None and ans.status == "ok" else {}
    changed = bool(s.get("p8_object_changed")) if "p8_object_changed" in s else None
    ev = {"object": obj, "frame_t": float(best.t), "blur": float(best.blur), "answer": getattr(ans, "value", None),
          "not_identifiable_prob": probs.get("not_identifiable"), "annotator_changed_object": changed}
    if not probs:
        return [_f("34", "row34", u, "unassessable", "No frame answer.", ev)]
    if probs.get("not_identifiable", 0) >= ctx.cfg.image_flag_prob:
        return [_f("34", "row34", u, "candidate", f'Specific noun "{obj}" but the object cannot be identified in the '
                   f"sharpest frame (p={probs['not_identifiable']:.2f}). A fallback noun or IDLE may be needed.", ev)]
    return [_f("34", "row34", u, "pass", f"Object identifiable ({ev['answer']}).", ev)]


register(CheckSpec(key="row34", rows="34", title="Confident guess instead of a fallback", category=CATEGORY,
                   uses=["P2", "P5", "P6", "P8"], units=lambda c: _units(c, _specific_object), run=_run_row34, workers=8))


# Row 28: intent instead of what happened (rides on rows 30 and 33) -------------------------------------

def _run_row28(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    got = {}
    for key in ("row30", "row33"):
        f = ctx.results.get(key)
        hit = f[f.segment_id == sid] if f is not None else pd.DataFrame()
        got[key] = (hit.status.iloc[0], hit.message.iloc[0]) if len(hit) else ("not run", "")
    ev = {"row30": got["row30"][0], "row33": got["row33"][0], "review_sample": _review_sample(ctx, "row28", sid, 0.05)}
    flagged = []
    if got["row30"][0] == "candidate":
        flagged.append(f"row 30: {got['row30'][1]}")
    if got["row33"][0] == "candidate" and "result never happens" in got["row33"][1]:
        flagged.append(f"row 33: {got['row33'][1]}")
    if flagged:
        return [_f("28", "row28", u, "candidate", "Describes what was meant, not what happened. " + " / ".join(flagged), ev)]
    if all(st in ("not run", "unassessable") for st, _ in got.values()):
        return [_f("28", "row28", u, "unassessable", "Run rows 30 and 33 first.", ev)]
    return [_f("28", "row28", u, "pass", "In the 5 percent review sample." if ev["review_sample"] else "", ev)]


register(CheckSpec(key="row28", rows="28", title="Intent instead of what happened", category=CATEGORY,
                   uses=["P5"], units=lambda c: _units(c, lambda x, s: not _literal(x, s)), run=_run_row28))


FAITHFULNESS_CHECKS = ["row29", "row30", "row31", "row32", "row33", "row34", "row28"]
