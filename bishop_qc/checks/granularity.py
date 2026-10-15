"""Segmentation errors / Granularity: sheet rows 8 to 17.

row08  Two actions merged into one segment.
row09  One action split into several segments.
row10  A micromovement folded into the next action (needs a gold set).
row11  A failed and a successful attempt left in one segment when the restart is clear.
row12  Attempts split when they run as one continuous motion.
row13  An exemption split anyway: walking with a hand action, or both hands at once.
row14  Separate pick and place cycles grouped into one segment.
row15  A regrip, small adjustment or settling hand treated as a new action.
row16  A clean hand to hand transfer outside a walk not recorded.
row17  An aborted reach that does not run to the withdrawal of the hand.

Text signals come from P5. Frame signals come from P3, P4 and P6 when they ran. Questions on one or
two frames go to OpenAI Decisions. Questions on a clip go to the model set per check in
cfg.clip_model: Sol for row 8, OpenAI Decisions (gpt-6-luna) for rows 11, 12 and 14. The notebook
shows the comparison behind that choice. Questions on caption text go to Jev.
Every flag is a candidate for review.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import lexicon as L
from ..clients import Question
from ..schema import Finding
from .boundary import _describe, _main_clause, _same_action, _sides, boundary_units, segment_units
from .runner import CheckContext, CheckSpec, Unit, register

CATEGORY = "Segmentation errors / Granularity"
NUMBER_WORDS = {"two", "three", "four", "five", "six", "several", "many", "both"}


# Shared helpers ------------------------------------------------------------------

def _seg(ctx, sid) -> pd.Series:
    return ctx.prims.segments.set_index("segment_id").loc[sid]


def _actions(ctx, sid):
    return [c for c in ctx.prims.parsed[sid].clauses if not c.is_holding]


def _is_literal(ctx, sid) -> bool:
    return ctx.prims.parsed[sid].is_fixed_literal


def _is_hand_action(ctx, sid) -> bool:
    c = _main_clause(ctx, sid)
    return c is not None and not _is_literal(ctx, sid) and not c.is_navigation


def _same_object(c1, c2) -> bool:
    if c1 is None or c2 is None or not c1.object_head or c1.object_head != c2.object_head:
        return False
    a1, a2 = set(c1.attributes), set(c2.attributes)
    return not a1 or not a2 or a1 == a2


def _plural(c) -> bool:
    head = (c.object_head or "").lower()
    return (head.endswith("s") and not head.endswith("ss")) or bool(set(c.attributes) & NUMBER_WORDS)


def clip_frames(ctx, rid: str, start: float, end: float, n: int = 6) -> List[str]:
    """Up to n timeline frames spread over [start, end], at most 2 per second."""
    fr = ctx.prims.frames
    f = fr[(fr.recording_id == rid) & fr.is_timeline & (fr.t >= start) & (fr.t <= end)].sort_values("t")
    if f.empty:
        return []
    k = int(min(n, len(f), max(2, np.floor((end - start) * 2) + 1)))
    idx = np.linspace(0, len(f) - 1, k).round().astype(int)
    return list(f.path.iloc[np.unique(idx)])


def frame_at(ctx, rid: str, t: float) -> Optional[str]:
    fr = ctx.prims.frames
    f = fr[fr.recording_id == rid]
    if f.empty:
        return None
    return f.path.iloc[int(np.argmin((f.t - t).abs().to_numpy()))]


def clip_model(ctx, key: str) -> str:
    m = ctx.cfg.clip_model
    return m.get(key, "decisions") if isinstance(m, dict) else m


def clip_question(ctx, key: str, q: Question, rid: str, start: float, end: float, context: str, n: int = 6):
    """Ask one closed question about a clip, sent as up to n frames in time order."""
    paths = clip_frames(ctx, rid, start, end, n)
    if not paths:
        return None, []
    step = (end - start) / max(len(paths) - 1, 1)
    state = (f"{context}\nThe {len(paths)} images are frames of a first person video clip, in time order, "
             f"from {start:.1f} s to {end:.1f} s, about {step:.1f} s apart.")
    if clip_model(ctx, key) == "sol":
        return ctx.clients.sol.ask(q, state, paths), paths
    return ctx.clients.decisions.ask([q], state, paths)[q.name], paths


def _prob(a) -> Optional[float]:
    return a.prob if a is not None and a.status == "ok" else None


def _contact_changes(ctx, rid, start, end, sides) -> Optional[Dict[str, int]]:
    ht = ctx.prims.hand_timeline
    if not len(ht):
        return None
    h = ht[(ht.recording_id == rid) & (ht.t >= start) & (ht.t < end) & ht.side.isin(sides)].sort_values("t")
    out = {}
    for side, g in h.groupby("side"):
        s = g.state[g.state.isin(["contact", "no_contact"])]
        out[side] = int((s != s.shift()).sum() - 1) if len(s) else 0
    return out


def _f(ctx_row: str, key: str, u: Unit, status: str, message: str, evidence: Dict) -> Finding:
    return Finding(row=ctx_row, check=key, recording_id=u["recording_id"], segment_id=u["segment_id"],
                   status=status, message=message, evidence=evidence)


def _rng(ctx, key: str, sid: str) -> random.Random:
    return random.Random(f"{ctx.cfg.seed}:{key}:{sid}")


# Row 8: two actions merged into one segment -----------------------------------------------

def _run_row08(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    if _is_literal(ctx, sid):
        return [_f("8", "row08", u, "pass", "Fixed caption.", {})]
    seg = _seg(ctx, sid)
    acts = _actions(ctx, sid)
    # "then" joins actions in sequence. "and" between different hands is the two hands exemption.
    seq = [c for c in acts[1:] if c.connector == "then"
           or (c.connector == "and" and (not c.actor or c.actor == acts[0].actor))]
    failure_tail = [c for c in acts[1:] if c.connector == "but" and c.has_failure_word]
    text_mark = bool(seq)
    long_mark = float(seg.get("p7_duration_pct_verb", 0) or 0) >= 0.8
    changes = _contact_changes(ctx, u["recording_id"], seg.start_s, seg.end_s, ["left", "right"])
    frame_mark = changes is not None and max(changes.values(), default=0) >= 3
    audit = (not text_mark and not long_mark and len(acts) == 1 and _rng(ctx, "row08", sid).random() < 0.10)
    ev = {"clauses": len(acts), "text_mark": text_mark, "long_mark": long_mark, "contact_changes": changes,
          "failure_tail": bool(failure_tail), "audit_sample": audit}
    if not (text_mark or long_mark or frame_mark or audit):
        return [_f("8", "row08", u, "pass", "One action in the caption, normal length.", ev)]
    q = Question("two_actions", "noul",
                 "Does this clip show two or more separate actions one after another, with a clear moment "
                 "where the next action starts? Holding an object while walking, both hands acting at once, "
                 "and repeating one motion on one object (for example shaking a blanket) each count as one action.")
    ans, _ = clip_question(ctx, "row08", q, u["recording_id"], seg.start_s, seg.end_s, f'The caption of this clip is "{seg.caption}".')
    p = _prob(ans)
    ev.update({"two_actions_prob": p, "source": getattr(ans, "source", "")})
    pf = ctx.cfg.image_flag_prob
    if text_mark and (p is None or p >= 0.5):
        why = "the caption joins actions with 'then'" + (f"; frames p={p:.2f}" if p is not None else "; frames not checked")
        return [_f("8", "row08", u, "candidate", f"Two actions in one segment: {why}.", ev)]
    if (long_mark or frame_mark or audit) and p is not None and p >= pf:
        return [_f("8", "row08", u, "candidate", f"The clip shows separate actions (p={p:.2f}).", ev)]
    if p is None and (long_mark or frame_mark):
        return [_f("8", "row08", u, "unassessable", "Long segment, but no frame answer.", ev)]
    msg = "Text joins actions but the clip looks continuous. Review the text." if text_mark else "Looks like one action."
    return [_f("8", "row08", u, "report" if text_mark else "pass", msg, ev)]


register(CheckSpec(key="row08", rows="8", title="Two actions merged into one segment", category=CATEGORY,
                   uses=["P2", "P3", "P5", "P7"], units=segment_units, run=_run_row08, workers=8))


# Row 9: one action split into several segments ----------------------------------------------

def _run_row09(ctx: CheckContext, u: Unit) -> List[Finding]:
    c1, c2 = _main_clause(ctx, u["prev_id"]), _main_clause(ctx, u["segment_id"])
    cont = u["next_caption"].lower().startswith("continue")
    mark = cont or (c1 is not None and c2 is not None and c1.verb_key == c2.verb_key and _same_object(c1, c2))
    ev = {"prev_id": u["prev_id"], "t": u["t"], "continue_word": cont}
    if not mark:
        return [_f("9", "row09", u, "pass", "Different verb or object on each side.", ev)]
    rid, t = u["recording_id"], u["t"]
    a, b = frame_at(ctx, rid, t - 0.3), frame_at(ctx, rid, t + 0.3)
    q = Question("one_or_two", "choice",
                 f"Two frames from a first person video, 0.6 s apart, around a cut between "
                 f"{_describe(u['prev_caption'], 'prev')} and {_describe(u['next_caption'], 'next')}. "
                 "Do the frames show one action still going on, or does one action end and a new one start between them?",
                 {"one_action": "The same single action is still going on, with no new action starting.",
                  "new_action": "One action ends and a new one starts between the frames, for example a new object is picked.",
                  "unclear": "The frames do not show it."})
    ans = ctx.clients.decisions.ask([q], "Frame 1 is 0.3 s before the cut. Frame 2 is 0.3 s after it.",
                                    [x for x in (a, b) if x])[q.name] if a and b else None
    probs = (ans.probs or {}) if ans is not None and ans.status == "ok" else {}
    ev.update({"one_action_prob": probs.get("one_action"), "new_action_prob": probs.get("new_action"),
               "source": getattr(ans, "source", "")})
    if not probs:
        return [_f("9", "row09", u, "unassessable", "Same verb and object, but no frame answer.", ev)]
    if probs.get("one_action", 0) >= ctx.cfg.image_flag_prob:
        return [_f("9", "row09", u, "candidate",
                   f"Same verb and object on both sides, and the frames show one continuing action "
                   f"(p={probs['one_action']:.2f}). Split may be inside one action.", ev)]
    return [_f("9", "row09", u, "pass", "Same verb and object, but a new action starts at the cut.", ev)]


register(CheckSpec(key="row09", rows="9", title="One action split into several segments", category=CATEGORY,
                   uses=["P2", "P5"], units=boundary_units, run=_run_row09, workers=8, unit="boundary"))


# Row 10: micromovement folded (gold set only) ----------------------------------------------

def _recording_units(ctx: CheckContext) -> Iterable[Unit]:
    for rid in ctx.prims.recordings.recording_id:
        yield {"recording_id": rid, "segment_id": ""}


def _run_row10(ctx: CheckContext, u: Unit) -> List[Finding]:
    gold_path = Path(ctx.cfg.data_dir) / "gold_segments.csv"
    if not gold_path.exists():
        return [_f("10", "row10", u, "no_coverage",
                   "A folded micromovement has no caption. Only a gold set can show it (gold_segments.csv).", {})]
    gold = pd.read_csv(gold_path)
    gold = gold[gold.recording_id == u["recording_id"]]
    seg = ctx.prims.segments
    starts = seg[seg.recording_id == u["recording_id"]].start_s.to_numpy()
    out = []
    for g in gold.itertuples():
        near = np.abs(starts - g.start_s).min() if len(starts) else np.inf
        if near > 0.2:
            out.append(Finding(row="10", check="row10", recording_id=u["recording_id"], segment_id="",
                               status="candidate", message=f"Gold segment at {g.start_s:.2f} s has no annotator "
                               f"boundary within 0.2 s (nearest {near:.2f} s).", evidence={"gold_start_s": g.start_s}))
    return out or [_f("10", "row10", u, "pass", "Every gold start has an annotator boundary within 0.2 s.", {})]


register(CheckSpec(key="row10", rows="10", title="Micromovement folded into the next action", category=CATEGORY,
                   uses=[], units=_recording_units, run=_run_row10, unit="recording"))


# Row 11: failed and successful attempts left together ---------------------------------------

def _run_row11(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    if not _is_hand_action(ctx, sid):
        return [_f("11", "row11", u, "pass", "Not a hand action.", {})]
    seg = _seg(ctx, sid)
    pc = ctx.prims.parsed[sid]
    fail_word = any(c.has_failure_word for c in pc.clauses) or bool(L.INTENT_RE.search(seg.caption))
    long_mark = float(seg.get("p7_duration_pct_verb", 0) or 0) > 0.5
    sampled = _rng(ctx, "row11", sid).random() < 0.10
    ev = {"failure_word": fail_word, "long_mark": long_mark, "random_sample": sampled}
    if not (fail_word or long_mark or sampled):
        return [_f("11", "row11", u, "pass", "Short segment with no failure word, not sampled.", ev)]
    q = Question("fail_then_retry", "noul",
                 "Does this clip show an attempt that fails (a miss, a slip, or a reach pulled back) and then a "
                 "new attempt at the same thing? Repeated small adjustments that gradually work an object into "
                 "place are one continuous attempt, not a failure and a retry.")
    ans, _ = clip_question(ctx, "row11", q, u["recording_id"], seg.start_s, seg.end_s, f'The caption of this clip is "{seg.caption}".', n=8)
    p = _prob(ans)
    ev.update({"fail_then_retry_prob": p, "source": getattr(ans, "source", "")})
    if p is None:
        return [_f("11", "row11", u, "unassessable", "No clip answer.", ev)]
    if p >= ctx.cfg.image_flag_prob:
        loc = "Needs the hand timeline (P3) to place the split frame." if not len(ctx.prims.hand_timeline) else ""
        return [_f("11", "row11", u, "candidate",
                   f"A failed attempt and a retry in one segment (p={p:.2f}). {loc}".strip(), ev)]
    return [_f("11", "row11", u, "pass", "No failed attempt followed by a retry.", ev)]


register(CheckSpec(key="row11", rows="11", title="Failed and successful attempts left together", category=CATEGORY,
                   uses=["P2", "P3", "P5", "P7"], units=segment_units, run=_run_row11, workers=8))


# Row 12: attempts split when they are one continuous motion ---------------------------------

def _run_row12(ctx: CheckContext, u: Unit) -> List[Finding]:
    p1, p2 = ctx.prims.parsed[u["prev_id"]], ctx.prims.parsed[u["segment_id"]]
    attempt = any(c.has_failure_word or c.intent for c in p1.clauses + p2.clauses)
    c1, c2 = _main_clause(ctx, u["prev_id"]), _main_clause(ctx, u["segment_id"])
    same_obj = _same_object(c1, c2)
    ev = {"prev_id": u["prev_id"], "t": u["t"], "attempt_word": attempt, "same_object": same_obj}
    if not (attempt and same_obj):
        return [_f("12", "row12", u, "pass", "No attempt on the same object on both sides.", ev)]
    s1, s2 = _seg(ctx, u["prev_id"]), _seg(ctx, u["segment_id"])
    q = Question("continuous", "noul",
                 "This clip was cut into two segments labelled as separate attempts. Do the attempts run as one "
                 "continuous motion, with no frame where a new attempt clearly begins?")
    ans, _ = clip_question(ctx, "row12", q, u["recording_id"], s1.start_s, s2.end_s,
                      f'Segment 1: "{s1.caption}". Segment 2 starts at {u["t"]:.1f} s: "{s2.caption}".', n=8)
    p = _prob(ans)
    ev.update({"continuous_prob": p, "source": getattr(ans, "source", "")})
    if p is None:
        return [_f("12", "row12", u, "unassessable", "No clip answer.", ev)]
    if p >= ctx.cfg.image_flag_prob:
        return [_f("12", "row12", u, "candidate", f"The attempts look like one continuous motion (p={p:.2f}).", ev)]
    if p > 1 - ctx.cfg.image_flag_prob:
        return [_f("12", "row12", u, "report", f"The model is unsure (p={p:.2f}). Kept as unsure.", ev)]
    return [_f("12", "row12", u, "pass", "A new attempt clearly begins at the cut.", ev)]


register(CheckSpec(key="row12", rows="12", title="Attempts split although continuous", category=CATEGORY,
                   uses=["P2", "P5"], units=boundary_units, run=_run_row12, workers=8, unit="boundary"))


# Row 13: an exemption split anyway ------------------------------------------------------------

def _span(mask: np.ndarray, t: np.ndarray) -> Optional[Tuple[float, float]]:
    return (float(t[mask].min()), float(t[mask].max())) if mask.any() else None


def _run_row13(ctx: CheckContext, u: Unit) -> List[Finding]:
    c1, c2 = _main_clause(ctx, u["prev_id"]), _main_clause(ctx, u["segment_id"])
    if c1 is None or c2 is None:
        return [_f("13", "row13", u, "pass", "A fixed caption on one side.", {})]
    nav_pair = c1.is_navigation != c2.is_navigation
    hands = {c1.actor, c2.actor}
    hand_pair = (not c1.is_navigation and not c2.is_navigation and hands == {"left hand", "right hand"})
    ev = {"prev_id": u["prev_id"], "t": u["t"], "nav_pair": nav_pair, "different_hands": hand_pair}
    if not (nav_pair or hand_pair):
        return [_f("13", "row13", u, "pass", "No exempt case on this cut.", ev)]
    ht, mo = ctx.prims.hand_timeline, ctx.prims.motion
    if not len(ht) or (nav_pair and not len(mo)):
        return [_f("13", "row13", u, "unassessable",
                   "Exempt case possible. Needs the hand timeline (P3) and the motion timeline (P4) to compare spans.", ev)]
    rid = u["recording_id"]
    s1, s2 = _seg(ctx, u["prev_id"]), _seg(ctx, u["segment_id"])
    lo, hi = float(s1.start_s), float(s2.end_s)
    h = ht[(ht.recording_id == rid) & (ht.t >= lo) & (ht.t < hi)]
    if nav_pair:
        m = mo[(mo.recording_id == rid) & (mo.track == "timeline") & (mo.t >= lo) & (mo.t < hi)].sort_values("t")
        walk = _span(m.forward_active.to_numpy(bool), m.t.to_numpy())
        hand_c = c2 if c1.is_navigation else c1
        side = hand_c.actor.split()[0] if hand_c.actor in ("left hand", "right hand") else None
        hh = h[h.side == side] if side else h
        act = _span((hh.state == "contact").to_numpy(), hh.t.to_numpy())
        ev.update({"walk_span": walk, "hand_span": act})
        if walk and act and min(walk[1], act[1]) - max(walk[0], act[0]) > 0.3:
            return [_f("13", "row13", u, "candidate",
                       "Walking and the hand action overlap in time, so they belong in one segment.", ev)]
        return [_f("13", "row13", u, "pass", "Walk and hand action do not overlap.", ev)]
    spans = {}
    for side in ("left", "right"):
        hh = h[h.side == side]
        spans[side] = _span((hh.state == "contact").to_numpy(), hh.t.to_numpy())
    ev["spans"] = spans
    a, b = spans["left"], spans["right"]
    if a and b and min(a[1], b[1]) - max(a[0], b[0]) > 0.3:
        return [_f("13", "row13", u, "candidate",
                   "Both hands act at the same time across the cut. Both hands at once is one segment.", ev)]
    return [_f("13", "row13", u, "pass", "The two hand actions do not overlap.", ev)]


register(CheckSpec(key="row13", rows="13", title="Exemption split anyway", category=CATEGORY,
                   uses=["P3", "P4", "P5"], units=boundary_units, run=_run_row13, unit="boundary"))


# Row 14: pick and place cycles grouped ----------------------------------------------------

def _run_row14(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    if not _is_hand_action(ctx, sid):
        return [_f("14", "row14", u, "pass", "Not a hand action.", {})]
    seg, c = _seg(ctx, sid), _main_clause(ctx, sid)
    plural = _plural(c) or bool(re.search(r"\b(each|all)\b", seg.caption, re.I))
    long_mark = float(seg.get("p7_duration_pct_verb", 0) or 0) >= 0.8
    ev = {"plural_object": plural, "long_mark": long_mark}
    if not (plural or long_mark):
        return [_f("14", "row14", u, "pass", "Single object, normal length.", ev)]
    q = Question("cycles", "choice",
                 "How many times does the person pick an object up and put it down in a different place in this "
                 "clip? Count each pick and place once. Shaking, wiping or washing one object without putting it "
                 "down counts as zero.",
                 {"zero": "No pick and place.", "one": "One pick and place.", "two": "Two.", "three_or_more": "Three or more."})
    ans, _ = clip_question(ctx, "row14", q, u["recording_id"], seg.start_s, seg.end_s, f'The caption of this clip is "{seg.caption}".', n=8)
    probs = (ans.probs or {}) if ans is not None and ans.status == "ok" else {}
    multi = probs.get("two", 0) + probs.get("three_or_more", 0) if probs else None
    ev.update({"cycles_answer": getattr(ans, "value", None), "two_or_more_prob": multi,
               "source": getattr(ans, "source", "")})
    if multi is None:
        return [_f("14", "row14", u, "unassessable", "No clip answer.", ev)]
    if multi >= ctx.cfg.image_flag_prob:
        return [_f("14", "row14", u, "candidate",
                   f"Two or more pick and place cycles in one segment (p={multi:.2f}). Each cycle is two segments.", ev)]
    return [_f("14", "row14", u, "pass", "At most one pick and place cycle.", ev)]


register(CheckSpec(key="row14", rows="14", title="Pick and place cycles grouped", category=CATEGORY,
                   uses=["P2", "P3", "P5", "P6", "P7"], units=segment_units, run=_run_row14, workers=8))


# Row 15: regrip or adjustment as a new action --------------------------------------------------

def _run_row15(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    if _is_literal(ctx, sid):
        return [_f("15", "row15", u, "pass", "Fixed caption.", {})]
    seg, c = _seg(ctx, sid), _main_clause(ctx, sid)
    segs = ctx.prims.segments
    g = segs[segs.recording_id == u["recording_id"]].sort_values("segment_index").reset_index(drop=True)
    i = int(g.index[g.segment_id == sid][0])
    neigh = [g.segment_id.iloc[j] for j in (i - 1, i + 1) if 0 <= j < len(g)]
    same_as = [n for n in neigh if _same_action(c, _main_clause(ctx, n))]
    short = float(seg.end_s - seg.start_s) < 0.5
    adjust = c is not None and c.verb in L.ADJUSTMENT_VERBS
    ev = {"short": short, "adjustment_verb": adjust, "same_action_as": same_as}
    if not (short or adjust or same_as):
        return [_f("15", "row15", u, "pass", "No adjustment signal.", ev)]
    ev["review_sample"] = _rng(ctx, "row15", sid).random() < 0.20
    if len(ctx.prims.hand_timeline):
        changes = _contact_changes(ctx, u["recording_id"], seg.start_s - 0.2, seg.end_s + 0.2, _sides(ctx, sid))
        ev["contact_changes"] = changes
    hints = [h for h, on in (("under 0.5 s", short), ("adjustment verb", adjust),
                             (f"same action as {', '.join(same_as)}", bool(same_as))) if on]
    return [_f("15", "row15", u, "candidate", "Possible regrip or adjustment cut as an action: " + "; ".join(hints)
               + (". In the 20 percent review sample." if ev["review_sample"] else "."), ev)]


register(CheckSpec(key="row15", rows="15", title="Regrip or adjustment as a new action", category=CATEGORY,
                   uses=["P3", "P5"], units=segment_units, run=_run_row15))


# Row 16: hand to hand transfer not recorded -----------------------------------------------------

def _run_row16(ctx: CheckContext, u: Unit) -> List[Finding]:
    c1, c2 = _main_clause(ctx, u["prev_id"]), _main_clause(ctx, u["segment_id"])
    if c1 is None or c2 is None:
        return [_f("16", "row16", u, "pass", "A fixed caption on one side.", {})]
    hands = {c1.actor, c2.actor}
    walk = c1.is_navigation or c2.is_navigation
    recorded = any("transfer" in x.lower() for x in (u["prev_caption"], u["next_caption"]))
    mark = _same_object(c1, c2) and hands == {"left hand", "right hand"} and not walk and not recorded
    ev = {"prev_id": u["prev_id"], "t": u["t"], "object": c1.object, "hands": sorted(hands)}
    if not mark:
        return [_f("16", "row16", u, "pass", "No change of hand on the same object outside a walk.", ev)]
    s1, s2 = _seg(ctx, u["prev_id"]), _seg(ctx, u["segment_id"])
    rid = u["recording_id"]
    a = frame_at(ctx, rid, float(s1.end_s) - 0.2)
    b = frame_at(ctx, rid, min(float(s2.start_s) + 0.6, (float(s2.start_s) + float(s2.end_s)) / 2))
    obj = c1.object.split(" of ")[0]
    q = Question("which_hand", "choice",
                 f"Two frames from a first person video, in time order. Look at the {obj}. "
                 "Which hand of the camera wearer holds it in each frame?",
                 {"moved_to_other_hand": f"The {obj} is in one hand in frame 1 and in the other hand in frame 2.",
                  "same_hand": f"The same hand holds the {obj} in both frames.",
                  "both_hands": f"Both hands hold the {obj} in at least one frame.",
                  "unclear": "Cannot tell."})
    ans = ctx.clients.decisions.ask([q], "Frame 1 is at the end of segment 1. Frame 2 is early in segment 2.",
                                    [a, b])[q.name] if a and b else None
    probs = (ans.probs or {}) if ans is not None and ans.status == "ok" else {}
    ev.update({"moved_prob": probs.get("moved_to_other_hand"), "answer": getattr(ans, "value", None),
               "source": getattr(ans, "source", "")})
    if not probs:
        return [_f("16", "row16", u, "unassessable", "Hand changes on the same object, but no frame answer.", ev)]
    if probs.get("moved_to_other_hand", 0) >= ctx.cfg.image_flag_prob:
        return [_f("16", "row16", u, "candidate",
                   f"The {obj} moves from one hand to the other (p={probs['moved_to_other_hand']:.2f}) and no segment "
                   "records the transfer.", ev)]
    return [_f("16", "row16", u, "report",
               f"Caption hands differ, but the frames do not show a transfer ({ev['answer']}). Check the hands (row 39).", ev)]


register(CheckSpec(key="row16", rows="16", title="Hand to hand transfer not recorded", category=CATEGORY,
                   uses=["P2", "P5"], units=boundary_units, run=_run_row16, workers=8, unit="boundary"))


# Row 17: aborted reach does not run to the withdrawal --------------------------------------------

def _run_row17(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    if not _is_hand_action(ctx, sid):
        return [_f("17", "row17", u, "pass", "Not a hand action.", {})]
    seg = _seg(ctx, sid)
    q = Question("failed_reach", "noul",
                 "Does this caption describe a reach or a grasp that failed, missed, or was pulled back before "
                 "the hand took the object?")
    ans = ctx.clients.jev.ask([q], {"caption": seg.caption})[q.name]
    p = _prob(ans)
    ev = {"failed_reach_prob": p, "source": getattr(ans, "source", "")}
    if p is None:
        return [_f("17", "row17", u, "unassessable", "No text answer.", ev)]
    if p < 0.5:
        return [_f("17", "row17", u, "pass", "Not a failed reach.", ev)]
    ht = ctx.prims.hand_timeline
    if not len(ht):
        return [_f("17", "row17", u, "unassessable",
                   f"Failed reach (Jev p={p:.2f}). Needs the hand timeline (P3) to see the withdrawal.", ev)]
    side = _sides(ctx, sid)
    end = float(seg.end_s)
    h = ht[(ht.recording_id == u["recording_id"]) & ht.side.isin(side) & (ht.t >= end - 0.5) & (ht.t <= end)
           & ht.detected].sort_values("t")
    if len(h) < 2:
        return [_f("17", "row17", u, "unassessable", "Hand not visible at the end of the segment.", ev)]
    # A hand moving away grows toward the bottom of the frame and shrinks in height.
    y_start, y_end = float(h.y1.iloc[0]), float(h.y1.iloc[-1])
    ev.update({"hand_top_start": y_start, "hand_top_end": y_end})
    if y_end <= y_start + 0.02:
        return [_f("17", "row17", u, "candidate", "The hand is still extended at the end of a failed reach. "
                   "The segment should run to the withdrawal.", ev)]
    return [_f("17", "row17", u, "pass", "The hand withdraws before the end.", ev)]


register(CheckSpec(key="row17", rows="17", title="Aborted reach cut before the withdrawal", category=CATEGORY,
                   uses=["P3", "P5"], units=segment_units, run=_run_row17, workers=8))


GRANULARITY_CHECKS = ["row08", "row09", "row10", "row11", "row12", "row13", "row14", "row15", "row16", "row17"]
