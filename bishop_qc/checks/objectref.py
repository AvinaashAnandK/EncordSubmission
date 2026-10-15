"""Caption errors / Object reference: sheet rows 58 to 71.

row58  The wrong object is named (a bowl called a plate).
row59  The object is not named specifically enough ("thing", "part", "piece").
row60  The reference does not single out one object among similar ones.
row61  Context words that need an earlier segment ("another", "again", "the last"). Covers row 62 too.
row63  The caption names an object by reading its label.
row64  The caption names the contents of a container that cannot be seen.
row65  A fallback noun is used when the object can be identified.
row66  A synonym is used in place of an approved fallback noun.
row67  A fallback noun with no attribute.
row68  An earlier caption edited once the object was identified in a later segment (platform fields).
row69  An object that no fallback noun fits (a guideline gap, report only).
row70  Over-detailed: more than one attribute beyond what singles the object out.
row71  Dialect words (tap and faucet), report only.

Text rules come first. Caption text questions go to Jev. Frame questions go to OpenAI Decisions, on a
SAM 3 crop of the object when P6 ran, otherwise on a whole frame with the object named. Sol suggests a
name for a vague or fallback noun, once per recording and phrase, for the reviewer. A model answer is a
candidate for review, never a rename.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import lexicon as L
from .. import parser as P
from ..clients import Question
from ..schema import Finding
from .boundary import segment_units
from .granularity import _f, _prob, _seg, clip_frames, frame_at
from .runner import CheckContext, CheckSpec, Unit, register

CATEGORY = "Caption errors / Object reference"
PRONOUNS = {"it", "them", "hands", "hand", "contents", "your hands"}


# Shared helpers ----------------------------------------------------------------------------

def _literal(ctx, sid) -> bool:
    return ctx.prims.parsed[sid].is_fixed_literal


def _phrases(ctx, sid) -> List[Tuple[str, str]]:
    """(role, phrase) for every object, place, held object and tool the caption names."""
    out: List[Tuple[str, str]] = []
    for c in ctx.prims.parsed[sid].clauses:
        for role, p in (("object", c.object), ("held", c.held_object), ("place", c.place),
                        ("from", c.place_from), ("tool", c.tool)):
            p = (p or "").strip().lower()
            if p and p not in PRONOUNS and p not in [x for _, x in out]:
                out.append((role, p))
    return out


def _head(phrase: str) -> str:
    return phrase.split(" of ")[0].split()[-1] if phrase else ""


def _of(phrase: str) -> str:
    return phrase.split(" of ", 1)[1] if " of " in phrase else ""


def _units(ctx, pred) -> Iterable[Unit]:
    for u in segment_units(ctx):
        if not _literal(ctx, u["segment_id"]) and pred(ctx, u["segment_id"]):
            yield u


def _timeline(ctx, rid, start, end) -> pd.DataFrame:
    fr = ctx.prims.frames
    return fr[(fr.recording_id == rid) & fr.is_timeline & (fr.t >= start) & (fr.t <= end)].sort_values("t")


def _sharpest(ctx, rid, start, end) -> Optional[pd.Series]:
    f = _timeline(ctx, rid, start, end)
    return None if f.empty else f.loc[f.blur.idxmax()]


def _blurred(ctx, rid, blur: float) -> bool:
    """Below the 10th percentile of the recording's timeline frames."""
    fr = ctx.prims.frames
    b = fr[(fr.recording_id == rid) & fr.is_timeline].blur
    return len(b) > 10 and blur < float(np.percentile(b, 10))


def _crop(ctx, sid, rid, phrase, frame: pd.Series) -> Tuple[str, str]:
    """A SAM 3 crop of the object on this frame when P6 has a box, else the whole frame."""
    ob = ctx.prims.objects
    prompt = phrase.split(" of ")[0]
    if len(ob):
        hit = ob[(ob.segment_id == sid) & (ob.prompt == prompt)]
        if len(hit):
            near = hit.iloc[(hit.t - frame.t).abs().argsort()[:5]]
            b = near.loc[near.score.idxmax()]
            fr = ctx.prims.frames
            src = fr[(fr.recording_id == rid) & (fr.frame_idx == b.frame_idx)]
            path = src.path.iloc[0] if len(src) else frame.path
            out = Path(ctx.cfg.cache_dir) / "crops" / f"{sid}_{re.sub(r'[^a-z0-9]+', '_', prompt)}_{int(b.frame_idx)}.jpg"
            if not out.exists():
                from PIL import Image

                out.parent.mkdir(parents=True, exist_ok=True)
                im = Image.open(path)
                w, h = im.size
                pw, ph = 0.15 * (b.x2 - b.x1), 0.15 * (b.y2 - b.y1)
                box = (max(0, (b.x1 - pw) * w), max(0, (b.y1 - ph) * h), min(w, (b.x2 + pw) * w), min(h, (b.y2 + ph) * h))
                im.crop(tuple(int(v) for v in box)).save(out, quality=90)
            return str(out), "sam3_crop"
    return frame.path, "whole_frame"


def _ask_frame(ctx, q: Question, path: str, how: str, phrase: str):
    where = (f'The image is a crop around "{phrase}" from a first person video.' if how == "sam3_crop"
             else f'The image is one frame of a first person video. The caption calls one object "{phrase}".')
    return ctx.clients.decisions.ask([q], where, [path]).get(q.name)


_NAME_LOCK = threading.Lock()


def _suggest_name(ctx, rid, phrase) -> Optional[str]:
    """Sol names the object once per recording and phrase, on the sharpest frame of its first segment."""
    seg = ctx.prims.segments
    hits = [r for r in seg[seg.recording_id == rid].sort_values("segment_index").itertuples()
            if phrase in [p for _, p in _phrases(ctx, r.segment_id)]]
    if not hits:
        return None
    fr = _sharpest(ctx, rid, float(hits[0].start_s), float(hits[0].end_s))
    if fr is None:
        return None
    schema = {"type": "object", "additionalProperties": False, "required": ["name", "identifiable"],
              "properties": {"name": {"type": "string"}, "identifiable": {"type": "boolean"}}}
    prompt = (f'One frame of a first person video. A caption calls one object "{phrase}". Give the most specific '
              "common noun phrase a person could use for that object from this frame, e.g., \"chicken nugget\" or "
              "\"plastic clip\". Set identifiable to false when nobody could tell what it is.")
    with _NAME_LOCK:
        data = ctx.clients.sol.json(prompt, schema, [fr.path], name="object_name")
    if not data:
        return None
    return data["name"] if data.get("identifiable") else f'not identifiable ("{data["name"]}" at best)'


def _choice(a) -> Dict[str, float]:
    return (a.probs or {}) if a is not None and a.status == "ok" else {}


# Row 58: wrong object named --------------------------------------------------------------------

def _confusable(head: str) -> List[str]:
    out = []
    for a, b in L.CONFUSABLE_PAIRS:
        if head == a and b not in out:
            out.append(b)
        elif head == b and a not in out:
            out.append(a)
    return out


def _has_confusable(ctx, sid) -> bool:
    return any(_confusable(_head(p)) for _, p in _phrases(ctx, sid))


def _run_row58(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid, rid = u["segment_id"], u["recording_id"]
    s = _seg(ctx, sid)
    best = _sharpest(ctx, rid, float(s.start_s), float(s.end_s))
    if best is None:
        return [_f("58", "row58", u, "unassessable", "No frames.", {})]
    ev, wrong, answered = {"frame_t": float(best.t), "blur": float(best.blur)}, [], False
    if _blurred(ctx, rid, float(best.blur)):
        return [_f("58", "row58", u, "unassessable", "The sharpest frame is blurred. Send to review.", ev)]
    for role, phrase in _phrases(ctx, sid):
        head, alts = _head(phrase), _confusable(_head(phrase))
        if not alts:
            continue
        opts = {head: f"The object is a {head}."}
        opts.update({a: f"The object is a {a}, not a {head}." for a in alts})
        opts.update({"other": "Something else.", "unclear": "Cannot see it well enough to tell."})
        q = Question(f"what_{head}", "choice", f'The caption calls this object "{phrase}". Which noun fits the object?', opts)
        path, how = _crop(ctx, sid, rid, phrase, best)
        probs = _choice(_ask_frame(ctx, q, path, how, phrase))
        ev[phrase] = {"probs": probs, "image": how}
        if probs:
            answered = True
            top = max(alts, key=lambda a: probs.get(a, 0))
            if probs.get(top, 0) >= ctx.cfg.image_flag_prob:
                wrong.append(f'"{phrase}" looks like a {top} (p={probs[top]:.2f})')
    if not answered:
        return [_f("58", "row58", u, "unassessable", "No frame answer.", ev)]
    if wrong:
        return [_f("58", "row58", u, "candidate", "Possibly the wrong noun: " + "; ".join(wrong) +
                   ". Do not rename without review.", ev)]
    return [_f("58", "row58", u, "pass", "The noun fits the object.", ev)]


register(CheckSpec(key="row58", rows="58", title="Wrong object named", category=CATEGORY,
                   uses=["P2", "P5", "P6"], units=lambda c: _units(c, _has_confusable), run=_run_row58, workers=8))


# Row 59: not specific enough ---------------------------------------------------------------------

def _vague(phrase: str) -> bool:
    return _head(phrase) in L.VAGUE_NOUNS or (_of(phrase) and _of(phrase).split()[-1] in L.VAGUE_NOUNS)


def _has_vague(ctx, sid) -> bool:
    return any(_vague(p) for _, p in _phrases(ctx, sid))


def _run_row59(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid, rid = u["segment_id"], u["recording_id"]
    hits = [p for _, p in _phrases(ctx, sid) if _vague(p)]
    names = {p: _suggest_name(ctx, rid, p) for p in hits}
    ev = {"vague": hits, "suggested_name": names}
    sugg = "; ".join(f'"{p}" (frame suggests: {n})' for p, n in names.items() if n)
    return [_f("59", "row59", u, "candidate", "Vague noun: " + ", ".join(f'"{p}"' for p in hits) +
               (". " + sugg if sugg else "") + ". Name the object, or use a fallback noun if it cannot be identified "
               "(row 69 logs objects no fallback noun fits).", ev)]


register(CheckSpec(key="row59", rows="59", title="Object not named specifically", category=CATEGORY,
                   uses=["P2", "P5"], units=lambda c: _units(c, _has_vague), run=_run_row59, workers=4))


# Row 60: reference does not single out one object -----------------------------------------------------

def _singular(phrase: str) -> bool:
    doc = P.nlp()("Pick up the " + phrase.split(" of ")[0])
    head = doc[-1]
    return head.tag_ != "NNS" and head.lemma_.lower() == head.text.lower()


def _singular_objects(ctx, sid) -> List[str]:
    return [p for role, p in _phrases(ctx, sid) if role in ("object", "held") and _singular(p)]


def _run_row60(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid, rid = u["segment_id"], u["recording_id"]
    s = _seg(ctx, sid)
    ob, ev, several, answered = ctx.prims.objects, {}, [], False
    f = _timeline(ctx, rid, float(s.start_s), float(s.end_s))
    if f.empty:
        return [_f("60", "row60", u, "unassessable", "No frames.", {})]
    first = f.iloc[min(2, len(f) - 1)]          # about 0.2 s in, before the hand moves the object
    mid = f.iloc[len(f) // 2]
    for phrase in _singular_objects(ctx, sid):
        if len(ob):
            hit = ob[(ob.segment_id == sid) & (ob.prompt == phrase.split(" of ")[0]) & (ob.score >= 0.5)]
            n = int(hit.groupby("frame_idx").obj_id.nunique().max()) if len(hit) else 0
            ev[phrase] = {"sam3_max_instances": n}
            answered = True
            if n >= 2:
                several.append(f'{n} objects match "{phrase}" in one frame (SAM 3)')
            continue
        q = Question("which_one", "choice", f'Can a reader tell from the caption which object "{phrase}" is?',
                     {"clear": "Only one object matches, or it is the one already in the hand, or the caption's "
                               "position, colour or place words single it out.",
                      "ambiguous": "Several matching objects are in view before the action, and nothing in the caption "
                                   'says which one is used, even with "a".',
                      "none": "No matching object is visible."})
        probs = _choice(ctx.clients.decisions.ask(
            [q], f'Caption: "{s.caption}". Two frames from the clip, at its start and its middle.',
            [first.path, mid.path]).get(q.name))
        ev[phrase] = {"probs": probs, "frame_t": float(first.t)}
        if probs:
            answered = True
            if probs.get("ambiguous", 0) >= ctx.cfg.image_flag_prob:
                several.append(f'several objects match "{phrase}" (p={probs["ambiguous"]:.2f})')
    if not answered:
        return [_f("60", "row60", u, "unassessable", "No frame answer.", ev)]
    if several:
        return [_f("60", "row60", u, "candidate", "The reference does not single out one object: " + "; ".join(several) +
                   ". Add an attribute that tells it apart (position, colour, size).", ev)]
    return [_f("60", "row60", u, "pass", "Each reference singles out one object.", ev)]


register(CheckSpec(key="row60", rows="60", title="Reference does not single out one object", category=CATEGORY,
                   uses=["P2", "P5", "P6"], units=lambda c: _units(c, lambda x, s: bool(_singular_objects(x, s))),
                   run=_run_row60, workers=8))


# Rows 61 and 62: context words -------------------------------------------------------------------------

def _context_words(caption: str) -> List[str]:
    out = []
    for m in L.CONTEXT_RE.finditer(caption):
        w = m.group(1).lower()
        if w not in out:
            out.append(w)
    return out


def _run_row61(ctx: CheckContext, u: Unit) -> List[Finding]:
    cap = str(_seg(ctx, u["segment_id"]).caption)
    words = _context_words(cap)
    probs = {}
    for w in words:
        q = Question("needs_earlier", "noul", f'In this caption, does the word "{w}" point to something outside the '
                     "caption, so that a reader would need an earlier caption or earlier footage to understand it? A word "
                     "that refers to something named earlier in the same caption is fine, e.g., \"it\" in \"Twist the cap "
                     "onto the bottle until it is closed\", and so is \"the other towel\" when two towels are in view.")
        probs[w] = _prob(ctx.clients.jev.ask([q], {"caption": cap, "word": w})[q.name])
    ev = {"words": words, "needs_earlier_prob": probs}
    if all(p is None for p in probs.values()):
        return [_f("61", "row61", u, "unassessable", "No text answer.", ev)]
    bad = [w for w, p in probs.items() if p is not None and p >= 0.5]
    if bad:
        return [_f("61", "row61", u, "candidate", "Needs an earlier segment: " + ", ".join(
            f'"{w}" (p={probs[w]:.2f})' for w in bad) + ". Each caption must stand alone.", ev)]
    return [_f("61", "row61", u, "pass", "Context words refer inside the caption.", ev)]


register(CheckSpec(key="row61", rows="61, 62", title="Caption depends on an earlier segment", category=CATEGORY,
                   uses=["P5"], units=lambda c: _units(c, lambda x, s: bool(_context_words(str(_seg(x, s).caption)))),
                   run=_run_row61, workers=8))


# Row 63: object named by reading its label -------------------------------------------------------------

def _run_row63(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid, rid = u["segment_id"], u["recording_id"]
    s = _seg(ctx, sid)
    q = Question("label_name", "noul", "Does this caption name an object by a brand, a product name, or contents that "
                 'a person could only know by reading a printed label? For example "the spinach bag" or "the Heinz '
                 'bottle". Plain descriptions by shape, colour or material ("the green bottle") are not.')
    p = _prob(ctx.clients.jev.ask([q], {"caption": str(s.caption)})[q.name])
    ev = {"label_name_prob": p}
    if p is None:
        return [_f("63", "row63", u, "unassessable", "No text answer.", ev)]
    if p < 0.5:
        return [_f("63", "row63", u, "pass", "No name taken from a label.", ev)]
    paths = clip_frames(ctx, rid, float(s.start_s), float(s.end_s), 4)
    if not paths:
        return [_f("63", "row63", u, "candidate", f"The caption may name an object from its label (p={p:.2f}). No frames "
                   "to check for printed text. The reviewer decides if the shape alone names it.", ev)]
    fq = Question("printed", "noul", "Is the object's name in the caption printed as text on the object in these "
                  "frames, so the caption reads the label instead of describing what the object looks like?")
    pf = _prob(ctx.clients.decisions.ask([fq], f'Caption: "{s.caption}". Frames in time order.', paths)[fq.name])
    ev["printed_prob"] = pf
    if pf is not None and pf >= ctx.cfg.image_flag_prob:
        return [_f("63", "row63", u, "candidate", f"The caption reads the label: the name is printed on the object "
                   f"(p={pf:.2f}). Describe it as seen, e.g., \"the white bag\".", ev)]
    return [_f("63", "row63", u, "pass", "The name is not read off a label.", ev)]


register(CheckSpec(key="row63", rows="63", title="Object named from its label", category=CATEGORY,
                   uses=["P2", "P5"], units=lambda c: _units(c, lambda x, s: True), run=_run_row63, workers=8))


# Row 64: contents of a container ---------------------------------------------------------------------------

def _contents(ctx, sid) -> List[Tuple[str, str]]:
    out = []
    for _, p in _phrases(ctx, sid):
        if _head(p) in L.CONTAINER_NOUNS and _of(p):
            out.append((p.split(" of ")[0], _of(p)))
    m = L.CONTENTS_RE.search(str(_seg(ctx, sid).caption))
    if m and not out:
        out.append(("container", str(_seg(ctx, sid).caption)[m.end():].strip()))
    return out


def _run_row64(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid, rid = u["segment_id"], u["recording_id"]
    s = _seg(ctx, sid)
    best = _sharpest(ctx, rid, float(s.start_s), float(s.end_s))
    if best is None:
        return [_f("64", "row64", u, "unassessable", "No frames.", {})]
    ev, bad, answered = {}, [], False
    for box, inside in _contents(ctx, sid):
        q = Question("contents", "choice", f'The caption says the {box} holds "{inside}". Can a person see that from the image?',
                     {"visible": f"The {box} is open or see through, and its contents can be seen to be {inside}.",
                      "not_visible": f"The {box} is closed or opaque. Nobody could tell what is inside by looking.",
                      "unclear": "Cannot tell."})
        path, how = _crop(ctx, sid, rid, box, best)
        probs = _choice(_ask_frame(ctx, q, path, how, f"{box} of {inside}"))
        ev[f"{box} of {inside}"] = {"probs": probs, "image": how}
        if probs:
            answered = True
            if probs.get("not_visible", 0) >= ctx.cfg.image_flag_prob:
                bad.append(f'"{inside}" cannot be seen in the {box} (p={probs["not_visible"]:.2f})')
    if not answered:
        return [_f("64", "row64", u, "unassessable", "No frame answer.", ev)]
    if bad:
        return [_f("64", "row64", u, "candidate", "Contents named but not visible: " + "; ".join(bad) +
                   ". Describe the container as seen.", ev)]
    return [_f("64", "row64", u, "pass", "The contents can be seen.", ev)]


register(CheckSpec(key="row64", rows="64", title="Contents of a container assumed", category=CATEGORY,
                   uses=["P2", "P5", "P6"], units=lambda c: _units(c, lambda x, s: bool(_contents(x, s))),
                   run=_run_row64, workers=8))


# Row 65: fallback noun on an identifiable object ---------------------------------------------------------------

def _fallbacks(ctx, sid) -> List[str]:
    return [p for _, p in _phrases(ctx, sid) if _head(p) in L.FALLBACK_NOUNS]


def _run_row65(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid, rid = u["segment_id"], u["recording_id"]
    s = _seg(ctx, sid)
    f = _timeline(ctx, rid, float(s.start_s), float(s.end_s))
    if f.empty:
        return [_f("65", "row65", u, "unassessable", "No frames.", {})]
    frames = f.iloc[np.unique(np.linspace(0, len(f) - 1, min(5, len(f))).round().astype(int))]
    ev, bad, answered = {}, [], False
    for phrase in _fallbacks(ctx, sid):
        noun = _head(phrase)
        q = Question("specific", "choice", f'The caption calls this object "{phrase}", using the fallback noun "{noun}". '
                     f"Can a person tell what specific kind of {noun} it is?",
                     {"specific": f"Yes. It is clearly a particular kind of {noun} with a common name, e.g., a jar or a bucket.",
                      "not_specific": f"No. It could be several kinds of {noun}, or it is too small, blurred or hidden.",
                      "unclear": "Cannot see it."})
        votes = []
        for fr in frames.itertuples():
            path, how = _crop(ctx, sid, rid, phrase, pd.Series(fr._asdict()))
            p = _choice(_ask_frame(ctx, q, path, how, phrase)).get("specific")
            if p is not None:
                votes.append(round(p, 2))
        ev[phrase] = {"specific_prob_per_frame": votes}
        if votes:
            answered = True
            confident = sum(v >= ctx.cfg.image_flag_prob for v in votes)
            if confident >= min(3, len(votes)):
                name = _suggest_name(ctx, rid, phrase)
                ev[phrase]["suggested_name"] = name
                bad.append(f'"{phrase}" looks identifiable in {confident} of {len(votes)} frames'
                           + (f" (suggested: {name})" if name else ""))
    if not answered:
        return [_f("65", "row65", u, "unassessable", "No frame answer.", ev)]
    if bad:
        return [_f("65", "row65", u, "candidate", "Fallback noun on an identifiable object: " + "; ".join(bad) +
                   '. The reviewer decides. "Uncertain" is a valid outcome.', ev)]
    return [_f("65", "row65", u, "pass", "The fallback noun fits: the object cannot be named more specifically.", ev)]


register(CheckSpec(key="row65", rows="65", title="Fallback noun on an identifiable object", category=CATEGORY,
                   uses=["P2", "P5", "P6"], units=lambda c: _units(c, lambda x, s: bool(_fallbacks(x, s))),
                   run=_run_row65, workers=8))


# Row 66: synonym in place of a fallback noun ---------------------------------------------------------------------

def _synonyms(ctx, sid) -> List[Tuple[str, str]]:
    return [(p, L.FALLBACK_SYNONYMS[_head(p)]) for _, p in _phrases(ctx, sid) if _head(p) in L.FALLBACK_SYNONYMS]


def _run_row66(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid, rid = u["segment_id"], u["recording_id"]
    s = _seg(ctx, sid)
    best = _sharpest(ctx, rid, float(s.start_s), float(s.end_s))
    ev, msgs = {}, []
    for phrase, fallback in _synonyms(ctx, sid):
        word = _head(phrase)
        p = None
        if best is not None:
            q = Question("identifiable", "noul", f'The caption calls this object "{phrase}". Can a person tell what the '
                         "object is, well enough to give it a specific common name?")
            path, how = _crop(ctx, sid, rid, phrase, best)
            p = _prob(_ask_frame(ctx, q, path, how, phrase))
        ev[phrase] = {"fallback": fallback, "identifiable_prob": p}
        if p is None:
            fix = f'use "{fallback}" if it cannot be identified, else its specific name'
        elif p >= ctx.cfg.image_flag_prob:
            fix = "the object looks identifiable, so give its specific name"
        else:
            fix = f'the object cannot be identified, so use the fallback noun "{fallback}"'
        msgs.append(f'"{word}" is not an approved fallback noun: {fix}')
    return [_f("66", "row66", u, "candidate", "; ".join(msgs) + ". The reviewer accepts or rejects the change.", ev)]


register(CheckSpec(key="row66", rows="66", title="Synonym in place of a fallback noun", category=CATEGORY,
                   uses=["P2", "P5"], units=lambda c: _units(c, lambda x, s: bool(_synonyms(x, s))),
                   run=_run_row66, workers=4))


# Row 67: fallback noun with no attribute ------------------------------------------------------------------------

def _attributes(phrase: str) -> List[str]:
    """Words before the head that describe it: adjectives, participles, numbers, colours."""
    doc = P.nlp()(phrase.split(" of ")[0])
    return [t.text for t in list(doc)[:-1] if t.pos_ in ("ADJ", "NUM", "VERB") or t.tag_ in ("VBN", "VBD", "JJ")]


def _run_row67(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    cap = str(_seg(ctx, sid).caption)
    bare, ev = [], {}
    for phrase in _fallbacks(ctx, sid):
        attrs = _attributes(phrase)
        if attrs:
            ev[phrase] = {"attributes": attrs}
            continue
        q = Question("has_attribute", "noul", f'Does the caption describe which {_head(phrase)} is meant, with a colour, '
                     f'size, material or position, before or after the noun? For example "the {_head(phrase)} on the left". '
                     'The place the object is taken from or put in ("from the counter", "in the sink") is part of the '
                     "action, not an attribute, so it does not count.")
        p = _prob(ctx.clients.jev.ask([q], {"caption": cap})[q.name])
        ev[phrase] = {"attributes": [], "has_attribute_prob": p}
        if p is None or p < 0.5:
            bare.append(phrase)
    if bare:
        return [_f("67", "row67", u, "candidate", "Fallback noun with no attribute: " + ", ".join(f'"{p}"' for p in bare) +
                   ". Add one that the frames support. Never invent one.", ev)]
    return [_f("67", "row67", u, "pass", "Each fallback noun has an attribute.", ev)]


register(CheckSpec(key="row67", rows="67", title="Fallback noun with no attribute", category=CATEGORY,
                   uses=["P5"], units=lambda c: _units(c, lambda x, s: bool(_fallbacks(x, s))), run=_run_row67, workers=4))


# Row 68: earlier caption edited after the object was identified later --------------------------------------------

def _run_row68(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid, rid = u["segment_id"], u["recording_id"]
    s = _seg(ctx, sid)
    flag = s.get("p9_backward_edit")
    if flag is None or pd.isna(flag):
        return [_f("68", "row68", u, "no_coverage", "No edit times for this caption. Ask the platform team for edit history.", {})]
    if not bool(flag):
        return [_f("68", "row68", u, "pass", "Not edited after later captions were created.", {})]
    seg = ctx.prims.segments
    later = seg[(seg.recording_id == rid) & (seg.segment_index > s.segment_index)]
    upd = pd.to_datetime(s.caption_updated_at, utc=True)
    before = later[pd.to_datetime(later.caption_created_at, utc=True) < upd]
    heads = {_head(p) for _, p in _phrases(ctx, sid)}
    shared = sorted({h for r in before.itertuples() for _, p in _phrases(ctx, r.segment_id) for h in [_head(p)]} & heads)
    ev = {"later_created_before_update": before.segment_id.tolist(), "edit_lag_s": s.get("p9_edit_lag_s"),
          "object_changed_from_suggestion": s.get("p8_object_changed"), "nearest_suggestion": s.get("p8_nearest_text"),
          "nouns_shared_with_those_segments": shared}
    why = []
    if bool(s.get("p8_object_changed")):
        why.append(f'the object differs from the nearest suggestion ("{s.get("p8_nearest_text")}")')
    if shared:
        why.append("it names " + ", ".join(shared) + ", also named in those later captions")
    return [_f("68", "row68", u, "candidate", f"Edited after {len(before)} later captions were created"
               + (": " + " and ".join(why) if why else "") + ". Check the noun against this segment's own footage.", ev)]


register(CheckSpec(key="row68", rows="68", title="Earlier caption edited after a later segment", category=CATEGORY,
                   uses=["P5", "P8", "P9"], units=lambda c: _units(c, lambda x, s: True), run=_run_row68))


# Row 69: no fallback noun fits (guideline gap) ------------------------------------------------------------------------

def _run_row69(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid, rid = u["segment_id"], u["recording_id"]
    s = _seg(ctx, sid)
    best = _sharpest(ctx, rid, float(s.start_s), float(s.end_s))
    if best is None:
        return [_f("69", "row69", u, "unassessable", "No frames.", {})]
    ev, gaps, answered = {}, [], False
    for phrase in [p for _, p in _phrases(ctx, sid) if _vague(p)]:
        opts = {"identifiable": "A person could give it a specific common name."}
        opts.update({n: f"It cannot be named specifically, but it is a {n}." for n in L.FALLBACK_NOUNS})
        opts["none_fits"] = ("It cannot be named specifically, and it is not a garment, container, utensil or linen "
                             "(e.g., a tool part, a toy, a gadget or a food).")
        q = Question("fallback_fit", "choice", f'The caption calls this object "{phrase}". Which describes it best?', opts)
        path, how = _crop(ctx, sid, rid, phrase, best)
        probs = _choice(_ask_frame(ctx, q, path, how, phrase))
        ev[phrase] = {"probs": probs}
        if probs:
            answered = True
            top = max(probs, key=probs.get)
            if top == "none_fits":
                gaps.append(f'"{phrase}" (p={probs[top]:.2f})')
    if not answered:
        return [_f("69", "row69", u, "unassessable", "No frame answer.", ev)]
    if gaps:
        return [_f("69", "row69", u, "report", "No fallback noun fits: " + ", ".join(gaps) +
                   ". A guideline gap for the client, not an annotator error.", ev)]
    return [_f("69", "row69", u, "pass", "The object can be named, or a fallback noun fits.", ev)]


register(CheckSpec(key="row69", rows="69", title="No fallback noun fits (guideline gap)", category=CATEGORY,
                   uses=["P2", "P5", "P6"], units=lambda c: _units(c, _has_vague), run=_run_row69, workers=8))


# Row 70: over-detailed ---------------------------------------------------------------------------------------

def _detailed(ctx, sid) -> List[Tuple[str, List[str]]]:
    return [(p, a) for role, p in _phrases(ctx, sid) if role in ("object", "held")
            for a in [_attributes(p)] if len(a) >= 2]


def _same_kind_elsewhere(ctx, rid, phrase) -> List[str]:
    """Other phrases in the recording with the same head noun, e.g., "small bag" for "large clear bag"."""
    seg = ctx.prims.segments
    head, base = _head(phrase), phrase.split(" of ")[0]
    return sorted({p.split(" of ")[0] for sid in seg[seg.recording_id == rid].segment_id for _, p in _phrases(ctx, sid)
                   if _head(p) == head and p.split(" of ")[0] != base} - {base})


def _run_row70(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid, rid = u["segment_id"], u["recording_id"]
    s = _seg(ctx, sid)
    f = _timeline(ctx, rid, float(s.start_s), float(s.end_s))
    if f.empty:
        return [_f("70", "row70", u, "unassessable", "No frames.", {})]
    first = f.iloc[min(2, len(f) - 1)]
    ev, over, answered = {}, [], False
    for phrase, attrs in _detailed(ctx, sid):
        head = _head(phrase)
        q = Question("how_many_kind", "choice", f"How many objects of the kind {head} are in this image?",
                     {"one": f"Only one {head}.", "several": f"Two or more {head}s.", "none": f"No {head} is visible."})
        probs = _choice(_ask_frame(ctx, q, first.path, "whole_frame", phrase))
        others = _same_kind_elsewhere(ctx, rid, phrase)
        ev[phrase] = {"attributes": attrs, "probs": probs, "same_kind_elsewhere": others}
        if not probs:
            continue
        answered = True
        top = max(probs, key=probs.get)
        # An attribute that tells it apart from a same-kind object named elsewhere in the recording is needed.
        needed = 0 if top == "one" and not others else 1
        if top != "none" and probs[top] >= ctx.cfg.image_flag_prob and len(attrs) > needed + 1:
            over.append(f'"{phrase}" has {len(attrs)} attributes ({", ".join(attrs)}) where {needed} single'
                        f'{"s" if needed == 1 else ""} it out ({top} {head} in view, no other {head} named in the recording)')
    if not answered:
        return [_f("70", "row70", u, "unassessable", "No frame answer.", ev)]
    if over:
        return [_f("70", "row70", u, "candidate", "Over-detailed (minor): " + "; ".join(over) +
                   ". One attribute beyond the minimum is allowed.", ev)]
    return [_f("70", "row70", u, "pass", "Attributes are within one of the minimum.", ev)]


register(CheckSpec(key="row70", rows="70", title="Over-detailed object description", category=CATEGORY,
                   uses=["P2", "P5", "P6"], units=lambda c: _units(c, lambda x, s: bool(_detailed(x, s))),
                   run=_run_row70, workers=8))


# Row 71: dialect words (report only) ----------------------------------------------------------------------------

DIALECT_RE = re.compile(r"\b(" + "|".join(re.escape(w) for pair in L.DIALECT_PAIRS.items() for w in pair) + r")s?\b", re.I)


def _recording_units(ctx) -> Iterable[Unit]:
    for rid in sorted(ctx.prims.segments.recording_id.unique()):
        yield {"recording_id": rid, "segment_id": ""}


def _run_row71(ctx: CheckContext, u: Unit) -> List[Finding]:
    seg = ctx.prims.segments
    caps = seg[seg.recording_id == u["recording_id"]].caption.astype(str)
    counts: Dict[str, int] = {}
    for c in caps:
        for m in DIALECT_RE.finditer(c):
            w = m.group(1).lower()
            counts[w] = counts.get(w, 0) + 1
    ev = {"counts": counts}
    if not counts:
        return [_f("71", "row71", u, "pass", "No dialect words.", ev)]
    pairs = {**L.DIALECT_PAIRS, **{v: k for k, v in L.DIALECT_PAIRS.items()}}
    return [_f("71", "row71", u, "report", "Dialect words: " + ", ".join(
        f'"{w}" in {n} captions (other form: "{pairs[w]}")' for w, n in counts.items()) +
        ". The guideline sets no dialect, so this is not scored.", ev)]


register(CheckSpec(key="row71", rows="71", title="Dialect words", category=CATEGORY, uses=["P5"],
                   units=_recording_units, run=_run_row71, unit="recording"))


OBJECT_REFERENCE_CHECKS = ["row58", "row59", "row60", "row61", "row63", "row64", "row65", "row66", "row67",
                           "row68", "row69", "row70", "row71"]
