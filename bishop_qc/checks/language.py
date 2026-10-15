"""Caption errors / Language and format: sheet rows 72 to 83.

row72  The caption is not a second-person imperative in the present tense.
row73  More than one sentence.
row74  A long compound sentence (over 20 words or over two connectors), report only.
row75  No capital letter at the start, or a period at the end.
row76  Spelling or grammar errors.
row77  US and UK spelling mixed in one recording, report only.
row78  A comma.
row79  A connector other than then, while, and, but, until, or a change of tense or person between clauses.
row80  A fixed caption mixed with other text.
row81  A fixed caption misspelled or in the wrong case.
row82  Different verbs for the same kind of action within a recording.
row83  A verb that departs from the house preferences, advice only.

Almost all of this is text. Rules come first: spaCy tags for mood and tense, regexes for format, SymSpell
for spelling. Jev confirms the rule hits that need judgement and answers the grammar question. Fixes for
format errors go in a proposed column. The original caption is never changed.
"""
from __future__ import annotations

import contextvars
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

from .. import lexicon as L
from .. import parser as P
from ..clients import Question
from ..schema import Finding
from .boundary import segment_units
from .granularity import _f, _prob, _seg
from .navigation import _ego_moves
from .objectref import _recording_units
from .runner import CheckContext, CheckSpec, Unit, register

CATEGORY = "Caption errors / Language and format"
BASE_TAGS = {"VB", "VBP"}          # "You pick" tags as VBP, the same form as the imperative


def _cap(ctx, sid) -> str:
    return str(_seg(ctx, sid).caption)


def _literal(ctx, sid) -> bool:
    return ctx.prims.parsed[sid].is_fixed_literal


def _text_units(ctx) -> Iterable[Unit]:
    for u in segment_units(ctx):
        if not _literal(ctx, u["segment_id"]):
            yield u


def _all_units(ctx) -> Iterable[Unit]:
    yield from segment_units(ctx)


def _jev(ctx, name, instructions, state) -> Optional[float]:
    q = Question(name, "noul", instructions)
    return _prob(ctx.clients.jev.ask([q], state)[q.name])


@lru_cache(maxsize=None)
def _verb_tag(clause_text: str) -> Tuple[str, str]:
    """(first word, its tag) of a clause read as "You <clause>", after any adverb."""
    doc = P._doc(clause_text)
    for t in list(doc)[1:]:
        if t.pos_ in ("ADV", "PUNCT") and t.tag_ not in ("VB", "VBP", "VBZ", "VBD", "VBG", "VBN"):
            continue
        return t.text, t.tag_
    return "", ""


def _clause_forms(ctx, sid) -> List[Dict]:
    """Mood and tense per clause, from spaCy tags. "while" clauses may use an -ing verb."""
    out = []
    for c in ctx.prims.parsed[sid].clauses:
        word, tag = _verb_tag(c.text)
        ok = tag in BASE_TAGS or (c.connector == "while" and tag == "VBG")
        out.append({"clause": c.text, "connector": c.connector, "word": word, "tag": tag, "imperative": ok})
    return out


# Row 72: not a second-person imperative ---------------------------------------------------------

def _run_row72(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    cap = _cap(ctx, sid)
    forms = _clause_forms(ctx, sid)
    first = forms[0] if forms else {"word": "", "tag": "", "imperative": False}
    starts_you = cap.lower().startswith("you ")
    ev = {"first_word": first["word"], "tag": first["tag"]}
    if first["imperative"] and not starts_you:
        return [_f("72", "row72", u, "pass", "Imperative, present tense.", ev)]
    if starts_you or first["tag"] in ("VBZ", "VBG"):
        # Unambiguous forms: "You pick", "Picks", "Picking". Past forms such as "put" or "set" go to Jev.
        return [_f("72", "row72", u, "candidate", f'Not an imperative: "{first["word"]}" ({first["tag"]}). Start with '
                   'the verb, e.g., "Pick up ...".', ev)]
    p = _jev(ctx, "imperative", 'Is the main clause of this caption a second-person imperative in the present tense, '
             'as if commanding a robot, e.g., "Pick up the cup with your right hand"? An adverb before the verb '
             '("Repeatedly pinch") is fine.', {"caption": cap})
    ev["imperative_prob"] = p
    if p is not None and p >= 0.5:
        return [_f("72", "row72", u, "pass", f"Imperative (Jev p={p:.2f}); the tagger read it as {first['tag'] or 'no verb'}.", ev)]
    return [_f("72", "row72", u, "candidate", f'Not an imperative: "{first["word"]}" reads as {first["tag"] or "no verb"}'
               + (f" (Jev p={p:.2f})" if p is not None else "") + '. Start with the verb, e.g., "Pick up ...".', ev)]


register(CheckSpec(key="row72", rows="72", title="Not a second-person imperative", category=CATEGORY,
                   uses=["P5"], units=lambda c: (u for u in _text_units(c) if not _mixed_literal(_cap(c, u["segment_id"]))),
                   run=_run_row72, workers=8))


# Row 73: more than one sentence ------------------------------------------------------------------

INNER_END_RE = re.compile(r"[.!?;](?=\s+[A-Za-z])")


def _run_row73(ctx: CheckContext, u: Unit) -> List[Finding]:
    cap = _cap(ctx, u["segment_id"])
    ends = INNER_END_RE.findall(cap)
    if not ends:
        return [_f("73", "row73", u, "pass", "One sentence.", {})]
    p = _jev(ctx, "several_sentences", "Is this caption more than one sentence? A period inside a name or a short "
             'form (e.g., "St.") does not end a sentence.', {"caption": cap})
    ev = {"inner_ends": ends, "several_sentences_prob": p}
    if p is not None and p < 0.5:
        return [_f("73", "row73", u, "pass", "The inner period does not end a sentence.", ev)]
    return [_f("73", "row73", u, "candidate", "More than one sentence. Two sentences usually mean two actions: see row 8.", ev)]


register(CheckSpec(key="row73", rows="73", title="More than one sentence", category=CATEGORY,
                   uses=["P5"], units=_text_units, run=_run_row73))


# Row 74: long compound sentence (report only) -----------------------------------------------------

def _run_row74(ctx: CheckContext, u: Unit) -> List[Finding]:
    cap = _cap(ctx, u["segment_id"])
    words = len(cap.split())
    conns = [m.group(1).lower() for m in P.CONNECTOR_RE.finditer(cap)]
    ev = {"words": words, "connectors": conns}
    if words > 20 or len(conns) > 2:
        why = [f"{words} words" if words > 20 else "", f"{len(conns)} connectors" if len(conns) > 2 else ""]
        return [_f("74", "row74", u, "report", "Long caption: " + " and ".join(w for w in why if w) +
                   ". Not scored: the guideline's towel caption has 22 words and is correct. Row 8 checks for a merge.", ev)]
    return [_f("74", "row74", u, "pass", "", ev)]


register(CheckSpec(key="row74", rows="74", title="Long compound sentence", category=CATEGORY,
                   uses=["P5"], units=_text_units, run=_run_row74))


# Row 75: capital and final period ----------------------------------------------------------------------

def _run_row75(ctx: CheckContext, u: Unit) -> List[Finding]:
    cap = _cap(ctx, u["segment_id"]).strip()
    problems = []
    if cap and not cap[0].isupper():
        problems.append("no capital letter at the start")
    if cap.endswith("."):
        problems.append("a period at the end")
    if not problems:
        return [_f("75", "row75", u, "pass", "", {})]
    fixed = (cap[:1].upper() + cap[1:]).rstrip(".").rstrip()
    return [_f("75", "row75", u, "candidate", "Format: " + " and ".join(problems) + f'. Proposed: "{fixed}".',
               {"proposed": fixed})]


register(CheckSpec(key="row75", rows="75", title="Capital letter and final period", category=CATEGORY,
                   uses=[], units=_text_units, run=_run_row75))


# Row 76: spelling and grammar ------------------------------------------------------------------------------

_SYM = {}


def _symspell():
    if "s" not in _SYM:
        from importlib.resources import files

        from symspellpy import SymSpell

        sym = SymSpell(max_dictionary_edit_distance=2)
        sym.load_dictionary(str(files("symspellpy") / "frequency_dictionary_en_82_765.txt"), 0, 1)
        _SYM["s"] = sym
    return _SYM["s"]


def project_words() -> set:
    """Words the general dictionary may lack: the lexicon's nouns, verbs, literals and both sides of the spelling maps."""
    words = set()
    for coll in (L.FALLBACK_NOUNS, L.VAGUE_NOUNS, L.CONTAINER_NOUNS, L.STATIC_FIXTURES, L.GREY_FURNITURE,
                 L.NAV_VERBS, L.DIRECTION_WORDS, L.FALLBACK_SYNONYMS, L.FIXED_LITERALS, L.FAILURE_WORDS,
                 L.UK_US_SPELLINGS.keys(), L.UK_US_SPELLINGS.values(), L.DIALECT_PAIRS.keys(), L.DIALECT_PAIRS.values()):
        for w in coll:
            words.update(re.findall(r"[a-z]+", str(w).lower()))
    return words


def _misspelled(cap: str) -> List[Tuple[str, str]]:
    from symspellpy import Verbosity

    sym, known = _symspell(), project_words()
    out = []
    for w in re.findall(r"[A-Za-z]+", cap):
        lw = w.lower()
        if len(lw) < 3 or lw in known or lw in sym.words:
            continue
        sugg = sym.lookup(lw, Verbosity.TOP, max_edit_distance=2)
        out.append((w, sugg[0].term if sugg else ""))
    return out


GRAMMAR_FLAG_PROB = 0.7      # grammar flags have low precision, so the bar is the same as for frame questions


def _run_row76(ctx: CheckContext, u: Unit) -> List[Finding]:
    cap = _cap(ctx, u["segment_id"])
    spelling = _misspelled(cap)
    p = _jev(ctx, "grammar_error", "Does this caption contain a grammar error, such as a wrong verb form, a missing "
             "word or a subject and verb that do not agree? Captions are imperatives with no final period, joined by "
             '"then", "while", "and" or "but". That style is correct and is not an error. Ignore a verb that is not an '
             "imperative and a change of tense or person between clauses: other checks own those.", {"caption": cap})
    ev = {"spelling": spelling, "grammar_error_prob": p}
    msgs = [f'"{w}"' + (f' (did you mean "{s}"?)' if s else "") for w, s in spelling]
    if p is not None and p >= GRAMMAR_FLAG_PROB:
        msgs.append(f"a grammar error (Jev p={p:.2f})")
    if msgs:
        return [_f("76", "row76", u, "candidate", "Spelling or grammar: " + "; ".join(msgs) +
                   ". Score it only when it changes the meaning.", ev)]
    if p is None:
        return [_f("76", "row76", u, "unassessable", "No grammar answer; spelling is clean.", ev)]
    return [_f("76", "row76", u, "pass", "", ev)]


register(CheckSpec(key="row76", rows="76", title="Spelling or grammar errors", category=CATEGORY,
                   uses=[], units=_text_units, run=_run_row76, workers=8))


# Row 77: mixed US and UK spelling (report only) ----------------------------------------------------------

def _run_row77(ctx: CheckContext, u: Unit) -> List[Finding]:
    seg = ctx.prims.segments
    text = " ".join(seg[seg.recording_id == u["recording_id"]].caption.astype(str)).lower()
    words = set(re.findall(r"[a-z]+", text))
    uk = sorted(w for w in L.UK_US_SPELLINGS if w in words)
    us = sorted(w for w in L.UK_US_SPELLINGS.values() if w in words)
    ev = {"uk": uk, "us": us}
    if uk and us:
        return [_f("77", "row77", u, "report", f"Mixed spelling: UK {', '.join(uk)} and US {', '.join(us)}. "
                   "Not scored until the client picks a locale.", ev)]
    std = "UK" if uk else "US" if us else "no marked"
    return [_f("77", "row77", u, "pass", f"One standard ({std} spelling).", ev)]


register(CheckSpec(key="row77", rows="77", title="Mixed US and UK spelling", category=CATEGORY,
                   uses=[], units=_recording_units, run=_run_row77, unit="recording"))


# Row 78: comma -----------------------------------------------------------------------------------------------

def _run_row78(ctx: CheckContext, u: Unit) -> List[Finding]:
    cap = _cap(ctx, u["segment_id"])
    if "," not in cap:
        return [_f("78", "row78", u, "pass", "", {})]
    fixed = re.sub(r"\s*,\s*(then|and|but|while)\b", r" \1", cap, flags=re.I)
    fixed = re.sub(r"\s*,\s*", " then ", fixed) if "," in fixed else fixed
    return [_f("78", "row78", u, "candidate", f'Warning: {cap.count(",")} comma(s). The guideline says to avoid them. '
               f'Proposed: "{fixed}". Check that "then" fits before accepting.', {"proposed": fixed})]


register(CheckSpec(key="row78", rows="78", title="Comma", category=CATEGORY, uses=[], units=_text_units, run=_run_row78))


# Row 79: other connectors, or a change of tense or person between clauses -------------------------------

OTHER_RE = re.compile(r"\b(" + "|".join(L.CONNECTORS_OTHER) + r")\b", re.I)


def _run_row79(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    cap = _cap(ctx, sid)
    others = sorted({m.group(1).lower() for m in OTHER_RE.finditer(cap)})
    forms = _clause_forms(ctx, sid)
    shifts = [f for f in forms[1:] if not f["imperative"]] if forms and forms[0]["imperative"] else []
    ev = {"other_connectors": others, "clauses": forms}
    msgs = []
    if others:
        msgs.append("connector " + ", ".join(f'"{w}"' for w in others) + ' (use then, while, and, but, or until)')
    if shifts:
        cap_q = "; ".join(f'"{f["clause"]}"' for f in shifts)
        p = _jev(ctx, "shift", "Does this caption change tense or person between its clauses, e.g., \"Pick up the cup "
                 'then places it on the table"? An "-ing" verb after "while" is fine.', {"caption": cap})
        ev["shift_prob"] = p
        if p is None or p >= 0.5:
            msgs.append(f"tense or person changes in {cap_q}" + (f" (Jev p={p:.2f})" if p is not None else ""))
    if msgs:
        return [_f("79", "row79", u, "candidate", "Clause joining: " + "; ".join(msgs) + ".", ev)]
    return [_f("79", "row79", u, "pass", "", ev)]


register(CheckSpec(key="row79", rows="79", title="Connector, tense or person change", category=CATEGORY,
                   uses=["P5"], units=lambda c: (u for u in _text_units(c) if len(c.prims.parsed[u["segment_id"]].clauses) > 1
                                                  or OTHER_RE.search(_cap(c, u["segment_id"]))),
                   run=_run_row79, workers=4))


# Rows 80 and 81: fixed captions ---------------------------------------------------------------------------------

LITERAL_RE = re.compile(r"\b(" + "|".join(L.FIXED_LITERALS) + r")\b")          # upper case only
LITERAL_START_RE = re.compile(r"^\s*(" + "|".join(l.replace(" ", r"[\s_\-]+") for l in L.FIXED_LITERALS) + r")\b(.+)$", re.I)


def _mixed_literal(cap: str) -> Optional[Tuple[str, str]]:
    if L.normalize_literal(cap) in L.FIXED_LITERALS:
        return None
    m = LITERAL_START_RE.match(cap)
    if m and m.group(2).strip(" .!?"):
        return re.sub(r"[\s_\-]+", " ", m.group(1)).upper(), m.group(2).strip()
    m = LITERAL_RE.search(cap)
    if m:
        return m.group(1), (cap[:m.start()] + cap[m.end():]).strip()
    return None


def _run_row80(ctx: CheckContext, u: Unit) -> List[Finding]:
    hit = _mixed_literal(_cap(ctx, u["segment_id"]))
    if not hit:
        return [_f("80", "row80", u, "pass", "", {})]
    lit, extra = hit
    return [_f("80", "row80", u, "candidate", f'The fixed caption {lit} is mixed with other text ("{extra}"). The extra '
               "words may describe a real action under a wrong label. Review the segment before any fix.",
               {"literal": lit, "extra": extra})]


register(CheckSpec(key="row80", rows="80", title="Fixed caption mixed with other text", category=CATEGORY,
                   uses=["P5"], units=_all_units, run=_run_row80))


def _near_literal(cap: str) -> Optional[Tuple[str, float]]:
    from rapidfuzz.distance import Levenshtein

    if len(cap.split()) > 3:
        return None
    best = min(L.FIXED_LITERALS, key=lambda l: Levenshtein.normalized_distance(cap.upper(), l))
    d = Levenshtein.normalized_distance(cap.upper(), best)
    return (best, d) if d <= 0.25 else None


def _run_row81(ctx: CheckContext, u: Unit) -> List[Finding]:
    cap = _cap(ctx, u["segment_id"])
    if cap in L.FIXED_LITERALS:
        return [_f("81", "row81", u, "pass", "Exact fixed caption.", {})]
    norm = L.normalize_literal(cap)
    if norm in L.FIXED_LITERALS:
        return [_f("81", "row81", u, "candidate", f'"{cap}" is the fixed caption {norm} in the wrong form. Proposed: "{norm}".',
                   {"proposed": norm})]
    near = _near_literal(cap)
    if near:
        return [_f("81", "row81", u, "candidate", f'"{cap}" is close to {near[0]} (distance {near[1]:.2f}). Review: a '
                   "near match can be a real action, so no fix is proposed.", {"near": near[0]})]
    return [_f("81", "row81", u, "pass", "", {})]


register(CheckSpec(key="row81", rows="81", title="Fixed caption misspelled or wrong case", category=CATEGORY,
                   uses=[], units=lambda c: (u for u in _all_units(c) if _literal(c, u["segment_id"])
                                             or L.normalize_literal(_cap(c, u["segment_id"])) in L.FIXED_LITERALS
                                             or _near_literal(_cap(c, u["segment_id"]))),
                   run=_run_row81))


# Row 82: verb consistency within a recording ---------------------------------------------------------------------

def _group(c) -> Optional[str]:
    vk = c.verb_key or c.verb
    if vk in ("put", "place", "set", "lay", "insert"):
        if c.place_prep in L.INTO_PREPS:
            return "put into" if vk in L.VERB_GROUPS["put into"] else None
        if c.place_prep in L.ONTO_PREPS:
            return "put onto" if vk in L.VERB_GROUPS["put onto"] else None
        return None
    for g, verbs in L.VERB_GROUPS.items():
        if g not in ("put into", "put onto") and vk in verbs:
            return g
    return None


def _run_row82(ctx: CheckContext, u: Unit) -> List[Finding]:
    rid = u["recording_id"]
    seg = ctx.prims.segments
    rows = seg[seg.recording_id == rid].sort_values("segment_index")
    uses: Dict[str, Dict[str, List[str]]] = {}
    who: Dict[str, set] = {}
    verbs = set()
    for r in rows.itertuples():
        for c in ctx.prims.parsed[r.segment_id].clauses:
            if c.is_holding or not c.verb:
                continue
            vk = c.verb_key or c.verb
            verbs.add(vk)
            g = _group(c)
            if g:
                uses.setdefault(g, {}).setdefault(vk, []).append(r.segment_id)
                who.setdefault(f"{g}:{vk}", set()).add(str(getattr(r, "annotator_id", "")))
    mixed = {g: {v: len(s) for v, s in vs.items()} for g, vs in uses.items() if len(vs) > 1}
    grouped = {v for vs in uses.values() for v in vs}
    loose = sorted(verbs - grouped)

    def ask(pair):
        a, b = pair
        return pair, _jev(ctx, "same_action", f'In egocentric video captions, do "{a}" and "{b}" describe the same kind '
                          "of action, so one recording should use only one of them? Different actions, such as "
                          '"wash" with a sponge and "rinse" under a tap, or "put into" and "place onto", are not the same.',
                          {"verb_a": a, "verb_b": b})

    pairs = list(combinations(loose, 2))
    with ThreadPoolExecutor(8) as pool:
        answers = list(pool.map(lambda pr: contextvars.copy_context().run(ask, pr), pairs))
    proposals = [f"{a} / {b} (p={p:.2f})" for (a, b), p in answers if p is not None and p >= 0.7]
    ev = {"mixed_groups": mixed, "annotators": {k: sorted(v) for k, v in who.items()}, "proposed_groups": proposals}
    if mixed:
        parts = [f'{g}: ' + ", ".join(f'"{v}" x{n}' for v, n in vs.items()) for g, vs in mixed.items()]
        return [_f("82", "row82", u, "candidate", "Different verbs for the same action: " + "; ".join(parts) +
                   ". Use one verb per kind of action in a recording." +
                   (f" Pairs Jev proposes for approval: {', '.join(proposals)}." if proposals else ""), ev)]
    if proposals:
        return [_f("82", "row82", u, "report", "No approved group is mixed. Pairs Jev proposes for a reviewer to approve: "
                   + ", ".join(proposals) + ".", ev)]
    return [_f("82", "row82", u, "pass", "One verb per kind of action.", ev)]


register(CheckSpec(key="row82", rows="82", title="Verb consistency within a recording", category=CATEGORY,
                   uses=["P5", "P9"], units=_recording_units, run=_run_row82, unit="recording"))


# Row 83: house verb preferences (advice only) ----------------------------------------------------------------------

def _run_row83(ctx: CheckContext, u: Unit) -> List[Finding]:
    sid = u["segment_id"]
    advice = []
    moves = None
    for c in ctx.prims.parsed[sid].clauses:
        vk = c.verb_key or c.verb
        if c.is_holding or not vk:
            continue
        if vk in L.HOUSE_VERB_MAP and L.HOUSE_VERB_MAP[vk] != "walk":
            advice.append(f'"{vk}" -> "{L.HOUSE_VERB_MAP[vk]}"')
        elif vk == "place" and c.place_prep in L.INTO_PREPS:
            advice.append('"place" into something -> "put" or "insert"')
        elif vk == "put" and c.place_prep in L.ONTO_PREPS:
            advice.append('"put" onto something -> "place"')
        elif vk in ("move", "go", "head", "proceed", "stroll"):
            moves = _ego_moves(ctx, sid) if moves is None else moves
            if any(m.text == c.text for m in moves):
                advice.append(f'"{vk}" for the ego moving -> "walk" ("move" is kept for objects)')
    if advice:
        return [_f("83", "row83", u, "report", "House verb advice: " + "; ".join(advice) + ". A preference, not a rule.",
                   {"advice": advice})]
    return [_f("83", "row83", u, "pass", "", {})]


register(CheckSpec(key="row83", rows="83", title="House verb preferences", category=CATEGORY,
                   uses=["P5"], units=_text_units, run=_run_row83, workers=4))


LANGUAGE_CHECKS = ["row72", "row73", "row74", "row75", "row76", "row77", "row78", "row79", "row80", "row81",
                   "row82", "row83"]
