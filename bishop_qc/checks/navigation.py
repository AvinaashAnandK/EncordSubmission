"""Caption errors / Navigation: sheet rows 50 to 57.

row50  Navigation has no landmark.
row51  A bare direction sequence ("go forward then turn right").
row52  A landmark that is not static (a small object or a person).
row53  Furniture other than the guideline's own examples used as a landmark (a client question, report only).
row54  Navigation described by the object the ego is about to use.
row55  A turn with no direction, or no end target; rotation with no turn in the caption (P4).
row56  "Turn around".
row57  A route that passes fixtures the caption does not name (review only).

Only clauses where the camera wearer moves count. "Turn on the tap" and "Move the fork to the right" move
an object, not the person. The parser decides first and Jev confirms each clause with a navigation verb.
"""
from __future__ import annotations

import re
from typing import Iterable, List

from .. import lexicon as L
from ..clients import Question
from ..schema import Finding
from .boundary import _main_clause, segment_units
from .granularity import _f, _prob, _seg, clip_frames
from .runner import CheckContext, CheckSpec, Unit, register

CATEGORY = "Caption errors / Navigation"
GUIDELINE_FURNITURE = {"sofa", "couch", "table", "coffee table", "shelf", "shelves"}   # used as landmarks on pages 6 and 9
FILLER = {"the", "a", "an", "to", "your", "you", "are", "until", "and", "then", "while", "but", "it", "its"}


def _ego_moves(ctx, sid) -> List:
    """Clauses where the camera wearer moves. The parser decides; Jev confirms clauses with a navigation verb."""
    pc = ctx.prims.parsed[sid]
    if pc.is_fixed_literal:
        return []
    out = []
    for c in pc.clauses:
        if c.is_holding or c.verb not in L.NAV_VERBS:
            continue
        q = Question("ego_moves", "noul", "In this clause, does the camera wearer move their own body through the "
                     "scene (walk, step or turn), rather than move or operate an object?")
        ans = ctx.clients.jev.ask([q], {"clause": c.text})[q.name]
        p = _prob(ans)
        if (p is not None and p >= 0.5) or (p is None and c.is_navigation):
            out.append(c)
    return out


def _nav_units(ctx) -> Iterable[Unit]:
    for u in segment_units(ctx):
        pc = ctx.prims.parsed[u["segment_id"]]
        if not pc.is_fixed_literal and any(c.verb in L.NAV_VERBS and not c.is_holding for c in pc.clauses):
            yield u


def _head(phrase: str) -> str:
    return phrase.split(" of ")[0].split()[-1] if phrase else ""


def _is_fixture(phrase: str) -> bool:
    p = phrase.lower()
    return p in L.STATIC_FIXTURES or _head(p) in L.STATIC_FIXTURES or p in GUIDELINE_FURNITURE


# Row 50: no landmark ----------------------------------------------------------------------------

def _run_row50(ctx: CheckContext, u: Unit) -> List[Finding]:
    moves = _ego_moves(ctx, u["segment_id"])
    if not moves:
        return [_f("50", "row50", u, "pass", "No clause where the camera wearer moves.", {})]
    bare = [c.text for c in moves if not c.landmarks and not c.facing]
    # "Turn right and walk past the island toward the sink": the walk gives the landmark for the turn.
    if bare and any(c.landmarks for c in moves):
        bare = [t for t in bare if not t.lower().startswith("turn")]
    ev = {"moves": [c.text for c in moves], "no_landmark": bare}
    if bare:
        return [_f("50", "row50", u, "candidate", "No landmark: " + "; ".join(f'"{t}"' for t in bare) +
                   ". Navigation is described against fixed landmarks.", ev)]
    return [_f("50", "row50", u, "pass", "Every movement has a landmark.", ev)]


register(CheckSpec(key="row50", rows="50", title="Navigation without a landmark", category=CATEGORY, uses=["P5"],
                   units=_nav_units, run=_run_row50))


# Row 51: bare direction sequence ------------------------------------------------------------------

def _run_row51(ctx: CheckContext, u: Unit) -> List[Finding]:
    moves = _ego_moves(ctx, u["segment_id"])
    if not moves:
        return [_f("51", "row51", u, "pass", "No clause where the camera wearer moves.", {})]
    cap = str(_seg(ctx, u["segment_id"]).caption).lower()
    words = re.findall(r"[a-z]+", cap)
    rest = [w for w in words if w not in L.NAV_VERBS | L.DIRECTION_WORDS | FILLER | set(L.CONNECTORS_ALLOWED)
            | {"go", "goes", "walking", "turning"}]
    ev = {"words_left": rest}
    if not rest:
        return [_f("51", "row51", u, "candidate", "A bare direction sequence: only verbs and directions. A reader cannot "
                   "trace the route.", ev)]
    return [_f("51", "row51", u, "pass", "", ev)]


register(CheckSpec(key="row51", rows="51", title="Bare direction sequence", category=CATEGORY, uses=["P5"],
                   units=_nav_units, run=_run_row51))


# Row 52: landmark not static ----------------------------------------------------------------------

LANDMARK_KIND = {"fixture": "A fixed part of a room: a wall, doorway, counter, sink, cabinet, stairs, or a large "
                            "appliance.",
                 "large_furniture": "Large furniture that normally stays in place: a sofa, table, bed or shelf.",
                 "small_object": "A small object that is easily moved: a cup, bag, sponge, box, chair or tool.",
                 "person": "A person or an animal."}


def _run_row52(ctx: CheckContext, u: Unit) -> List[Finding]:
    moves = _ego_moves(ctx, u["segment_id"])
    marks = [l for c in moves for l in c.landmarks]
    if not marks:
        return [_f("52", "row52", u, "pass", "No landmark to check.", {})]
    bad, unknown, kinds = [], [], {}
    for lm in marks:
        if _is_fixture(lm) or _head(lm) in L.GREY_FURNITURE:
            kinds[lm] = "fixture or furniture (list)"
            continue
        if _head(lm) in L.PERSON_WORDS:
            kinds[lm] = "person (list)"
            bad.append(lm)
            continue
        q = Question("landmark_kind", "choice", f'What kind of thing is "{lm}" in a home?', LANDMARK_KIND)
        a = ctx.clients.jev.ask([q], {"landmark": lm})[q.name]
        kinds[lm] = a.value if a.status == "ok" else "unknown"
        if a.status != "ok":
            unknown.append(lm)
        elif a.value in ("small_object", "person"):
            bad.append(lm)
    ev = {"landmarks": kinds}
    if bad:
        return [_f("52", "row52", u, "candidate", "Not a static landmark: " + ", ".join(bad) +
                   ". Use a fixed feature of the room.", ev)]
    if unknown:
        return [_f("52", "row52", u, "report", "Landmark not classified: " + ", ".join(unknown) + ". Added to the list "
                   "for weekly review.", ev)]
    return [_f("52", "row52", u, "pass", "Every landmark is static.", ev)]


register(CheckSpec(key="row52", rows="52", title="Landmark not static", category=CATEGORY, uses=["P5"],
                   units=_nav_units, run=_run_row52))


# Row 53: furniture outside the guideline's examples (client question) ---------------------------

def _run_row53(ctx: CheckContext, u: Unit) -> List[Finding]:
    moves = _ego_moves(ctx, u["segment_id"])
    grey = [l for c in moves for l in c.landmarks if _head(l) in L.GREY_FURNITURE and l not in GUIDELINE_FURNITURE]
    if grey:
        return [_f("53", "row53", u, "report", "Furniture as a landmark: " + ", ".join(grey) + ". The guideline uses a "
                   "sofa, table and shelf; where the line sits for other furniture is a client question. Counted, not "
                   "scored.", {"furniture": grey})]
    return [_f("53", "row53", u, "pass", "", {})]


register(CheckSpec(key="row53", rows="53", title="Furniture landmark (client question)", category=CATEGORY,
                   uses=["P5"], units=_nav_units, run=_run_row53))


# Row 54: navigation by the target object ----------------------------------------------------------

def _run_row54(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    moves = _ego_moves(ctx, sid)
    seg = ctx.prims.segments
    g = seg[seg.recording_id == u["recording_id"]].sort_values("segment_index")
    ids = list(g.segment_id)
    i = ids.index(sid)
    nxt = _main_clause(ctx, ids[i + 1]) if i + 1 < len(ids) else None
    # The next action may sit in a later clause of the same caption.
    pc = ctx.prims.parsed[sid]
    later = [c for c in pc.clauses if not c.is_holding and not c.is_navigation and c.object]
    targets = [c.object_head for c in later] + ([nxt.object_head] if nxt is not None and nxt.object_head else [])
    hits = [l for c in moves for l in c.landmarks if _head(l) in targets and not _is_fixture(l)]
    ev = {"landmarks": [l for c in moves for l in c.landmarks], "next_objects": targets}
    if hits:
        return [_f("54", "row54", u, "candidate", "Navigates by the object about to be used: " + ", ".join(hits) +
                   ". Name the fixture the ego heads toward instead.", ev)]
    return [_f("54", "row54", u, "pass", "", ev)]


register(CheckSpec(key="row54", rows="54", title="Navigation by the target object", category=CATEGORY, uses=["P5"],
                   units=_nav_units, run=_run_row54))


# Row 55: turn direction and end target ------------------------------------------------------------

def _run_row55(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    moves = _ego_moves(ctx, sid)
    turns = [c for c in moves if c.verb == "turn"]
    msgs = []
    for c in turns:
        k = moves.index(c)
        target = c.facing or c.landmarks or any(m.landmarks for m in moves[k + 1:])
        if c.direction not in ("left", "right"):
            msgs.append(f'"{c.text}" gives no direction (left or right)')
        if not target:
            msgs.append(f'"{c.text}" does not say what the ego ends up facing')
    ev = {"turns": [c.text for c in turns]}
    mo = ctx.prims.motion
    if len(mo) and moves and not turns:
        s = _seg(ctx, sid)
        m = mo[(mo.recording_id == u["recording_id"]) & (mo.track == "timeline") & (mo.t > s.start_s) & (mo.t <= s.end_s)]
        share = float(m.camera_turn_active.mean()) if len(m) else 0.0
        ev["turn_share"] = share
        if share >= 0.3:
            msgs.append("the camera turns for much of the segment, but the caption has no turn")
    if msgs:
        return [_f("55", "row55", u, "candidate", "Turn: " + "; ".join(msgs) + ".", ev)]
    return [_f("55", "row55", u, "pass", "", ev)]


register(CheckSpec(key="row55", rows="55", title="Turn without direction or end target", category=CATEGORY,
                   uses=["P4", "P5"], units=_nav_units, run=_run_row55))


# Row 56: "turn around" ------------------------------------------------------------------------------

def _run_row56(ctx: CheckContext, u: Unit) -> List[Finding]:
    cap = str(_seg(ctx, u["segment_id"]).caption)
    if re.search(r"\bturn(s|ed|ing)? around\b", cap, re.I):
        return [_f("56", "row56", u, "candidate", '"Turn around" is not a caption. Name the direction and what the ego '
                   'ends up facing, e.g., "Turn left until you are facing the doorway".', {})]
    return [_f("56", "row56", u, "pass", "", {})]


register(CheckSpec(key="row56", rows="56", title='"Turn around"', category=CATEGORY, uses=["P5"],
                   units=_nav_units, run=_run_row56))


# Row 57: route not traceable --------------------------------------------------------------------------

def _run_row57(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    moves = _ego_moves(ctx, sid)
    if not moves:
        return [_f("57", "row57", u, "pass", "No clause where the camera wearer moves.", {})]
    s = _seg(ctx, sid)
    if float(s.end_s - s.start_s) <= 3.0:
        return [_f("57", "row57", u, "pass", "Short walk; one landmark is enough.", {})]
    paths = clip_frames(ctx, u["recording_id"], float(s.start_s), float(s.end_s), 8)
    if not paths:
        return [_f("57", "row57", u, "unassessable", "No frames for this segment.", {})]
    named = [l for c in moves for l in c.landmarks]
    q = Question("passes_unnamed", "noul", "Does the camera wearer walk past fixed parts of the room (a counter, an "
                 "island, a doorway, a table) that the caption does not name, so that a reader could not trace the route?")
    a = ctx.clients.decisions.ask([q], f'Caption: "{s.caption}". Frames in time order.', paths)[q.name]
    p = _prob(a)
    ev = {"named_landmarks": named, "passes_unnamed_prob": p}
    if p is None:
        return [_f("57", "row57", u, "unassessable", "No frame answer.", ev)]
    if p >= ctx.cfg.image_flag_prob:
        return [_f("57", "row57", u, "candidate", f"The route passes fixtures the caption does not name (p={p:.2f}). "
                   "Review only: not every visible landmark must be named.", ev)]
    return [_f("57", "row57", u, "pass", "The route is traceable from the caption.", ev)]


register(CheckSpec(key="row57", rows="57", title="Route not traceable", category=CATEGORY, uses=["P2", "P5", "P6"],
                   units=_nav_units, run=_run_row57, workers=4))


NAVIGATION_CHECKS = ["row50", "row51", "row52", "row53", "row54", "row55", "row56", "row57"]
