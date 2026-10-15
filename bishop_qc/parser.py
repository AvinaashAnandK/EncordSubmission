"""Caption parser: rules plus spaCy, with Sol as the fallback.

Captions follow a narrow pattern, e.g., "Pick up the cup from the table with your right hand".
The rules handle the pattern. spaCy tags the words. When the rules cannot finish a clause and
Sol is available, Sol extracts the same fields with a fixed JSON schema.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional, Tuple

from . import lexicon as L
from .schema import Clause, ParsedCaption

_NLP = None
_MODEL = "en_core_web_sm"

SPLIT_RE = re.compile(r"\s+(then|but|while)\s+", re.I)
AND_RE = re.compile(r"\s+and\s+", re.I)
UNTIL_RE = re.compile(r"\s+until\s+.*$", re.I)
CONNECTOR_RE = re.compile(
    r"\b(" + "|".join(L.CONNECTORS_ALLOWED + L.CONNECTORS_OTHER) + r")\b", re.I
)
MULTI_PREPS = sorted({p for p in L.PLACE_TO_PREPS + L.PLACE_FROM_PREPS + L.LANDMARK_PREPS}, key=len, reverse=True)


def set_model(name: str) -> None:
    global _MODEL, _NLP
    if name != _MODEL:
        _MODEL, _NLP = name, None
        parse_caption.cache_clear()


def nlp():
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load(_MODEL, exclude=["ner"])
    return _NLP


def _doc(text: str):
    """Parse as "You <text>" so spaCy reads the imperative verb as a verb."""
    body = text[:1].lower() + text[1:] if text else text
    return nlp()("You " + body)


def _clean_np(tokens) -> Tuple[str, str, List[str], str]:
    """Return (phrase without determiner, determiner, attributes, head)."""
    toks = list(tokens)
    det = ""
    while len(toks) > 1 and (toks[0].pos_ in ("DET", "PRON") or toks[0].dep_ in ("det", "poss")):
        det = (det + " " + toks[0].text.lower()).strip()
        toks = toks[1:]
    if not toks:
        return "", det, [], ""
    head = toks[-1]
    attrs = [t.text.lower() for t in toks[:-1] if t.pos_ in ("ADJ", "NUM", "NOUN", "PROPN", "VERB", "ADV")]
    phrase = " ".join(t.text.lower() for t in toks).replace(" - ", "-")
    return phrase, det, attrs, head.text.lower()


NP_STOP = {p for p in MULTI_PREPS if " " not in p} | {
    "and", "then", "while", "but", "until", "together", "down", "up", "out", "away", "aside",
    "back", "over", "apart", "again", "upside", "with", "of"}
NP_STOP_POS = {"ADP", "CCONJ", "SCONJ", "PUNCT", "PART", "AUX"}
# An end state after the noun, e.g., "Push the cabinet door closed".
RESULT_WORDS = {"closed", "open", "shut", "flat", "upright", "straight", "clean", "dry", "full",
                "empty", "tight", "loose"}


class _Span:
    def __init__(self, doc, start: int, end: int):
        self.doc, self.start, self.end = doc, start, end

    def __iter__(self):
        return iter(self.doc[self.start:self.end])


def _np_after(doc, start: int):
    """Noun phrase that starts at token `start`: optional determiner, then words up to a stop.

    spaCy's own noun chunks break on phrases such as "red handled peeler", so the span is
    built here from the caption pattern instead.
    """
    i = start
    while i < len(doc) and (doc[i].pos_ in ("DET", "PRON") or doc[i].dep_ in ("det", "poss")) \
            and doc[i].text.lower() not in ("it", "them"):
        i += 1
    if start < len(doc) and doc[start].text.lower() in ("it", "them"):
        return _Span(doc, start, start + 1)
    j = i
    while j < len(doc):
        t = doc[j]
        low = t.text.lower()
        if t.text == "-" and j > i:          # "bottom-right corner"
            j += 1
            continue
        if low in NP_STOP or t.pos_ in NP_STOP_POS:
            break
        if low in RESULT_WORDS and j > i and doc[j - 1].pos_ in ("NOUN", "PROPN"):
            break
        j += 1
    if j == i:
        return None
    return _Span(doc, start, j)


def _match_prep(doc, i: int) -> Tuple[str, int]:
    """Longest preposition (one or more words) at token i. Returns (prep, n_tokens)."""
    words = [t.text.lower() for t in doc[i:i + 3]]
    for prep in MULTI_PREPS:
        parts = prep.split()
        if words[: len(parts)] == parts:
            return prep, len(parts)
    return "", 0


def _split(caption: str) -> List[Tuple[str, str]]:
    """Split into (connector, clause text). "and" splits only before a verb."""
    parts: List[Tuple[str, str]] = []
    pieces = SPLIT_RE.split(caption)
    conn = ""
    for i, piece in enumerate(pieces):
        if i % 2 == 1:
            conn = piece.lower()
            continue
        sub = AND_RE.split(piece)
        buf = sub[0]
        for nxt in sub[1:]:
            first = nxt.split()[0] if nxt.split() else ""
            if first and _doc(first + " it")[1].pos_ == "VERB":
                parts.append((conn, buf))
                conn, buf = "and", nxt
            else:
                buf = buf + " and " + nxt
        parts.append((conn, buf))
        conn = ""
    return [(c, t.strip()) for c, t in parts if t.strip()]


def _parse_clause(index: int, conn: str, text: str) -> Tuple[Clause, List[str]]:
    c = Clause(index=index, text=text, connector=conn)
    warn: List[str] = []
    work = text

    m = L.HOLDING_RE.search(work)
    if conn == "while" and work.lower().startswith("holding") and m:
        c.is_holding, c.verb, c.verb_key = True, "hold", "hold"
        c.held_object = re.sub(r"^(the|a|an|your)\s+", "", m.group("obj").lower())
        c.object, c.object_head = c.held_object, c.held_object.split()[-1]
        side = m.group("side").lower()
        c.held_actor = c.actor = "both hands" if side == "both" else f"{side} hand"
        return c, warn

    stop = UNTIL_RE.search(work)
    if stop:
        c.stop = stop.group(0).strip()
        f = L.FACING_RE.search(c.stop)
        if f:
            c.facing = f.group("target").strip().lower()
        work = work[: stop.start()]

    tool = L.TOOL_RE.search(work)
    body = L.BODY_RE.search(work)
    hand = L.HAND_RE.search(work)
    if tool:
        c.tool, c.actor = tool.group("tool").lower(), f"{tool.group('side').lower()} hand"
        work = work[: tool.start()] + work[tool.end():]
    elif body:
        c.actor = body.group("part").lower()
        work = work[: body.start()] + work[body.end():]
    elif hand:
        side = (hand.group("side") or hand.group("both") or "").lower()
        c.actor = "both hands" if side == "both" else f"{side} hand"
        work = work[: hand.start()] + work[hand.end():]
    d = L.DIRECTION_RE.search(work)
    if d:
        c.direction = (d.group("side") or d.group("updown")).lower()
        work = work[: d.start()] + work[d.end():]
    work = re.sub(r"\s{2,}", " ", work).strip()

    doc = _doc(work)
    i = 1
    adv = []
    while i < len(doc) and doc[i].pos_ == "ADV" and doc[i].text.lower() not in L.DIRECTION_WORDS:
        adv.append(doc[i].text)
        i += 1
    c.adverb = " ".join(adv)
    im = L.INTENT_RE.match(doc[i:].text) if i < len(doc) else None
    if im:
        c.intent = im.group(1).lower()
        i += len(c.intent.split())
    if i >= len(doc):
        warn.append("no verb")
        return c, warn
    vtok = doc[i]
    c.verb = (vtok.lemma_ if vtok.pos_ in ("VERB", "AUX") else vtok.text).lower()
    if vtok.pos_ != "VERB":
        warn.append(f"verb tagged {vtok.pos_}")
    i += 1
    if i < len(doc):
        nxt = doc[i].text.lower()
        if f"{c.verb} {nxt}" in L.PHRASAL_VERBS or (doc[i].dep_ == "prt" and nxt in L.PHRASAL_PARTICLES):
            c.particle = nxt
            i += 1
    c.verb_key = f"{c.verb} {c.particle}".strip()
    c.has_failure_word = c.verb in L.FAILURE_WORDS or bool(
        {t.text.lower() for t in doc[1:]} & L.FAILURE_WORDS)

    if c.verb == "turn" and i < len(doc) and doc[i].text.lower() in ("left", "right", "around"):
        c.direction = c.direction or doc[i].text.lower()
        i += 1
    # "fail to join them": keep "fail" as the verb and read the object of "join".
    if c.verb in L.FAILURE_WORDS and i + 1 < len(doc) and doc[i].text.lower() == "to":
        i += 2

    chunk = _np_after(doc, i)
    if chunk is not None and chunk.end - chunk.start == 1 and doc[chunk.start].text.lower() in ("it", "them"):
        c.object = c.object_head = doc[chunk.start].text.lower()
        i = chunk.end
        chunk = None
    if chunk is not None:
        c.object, c.determiner, c.attributes, c.object_head = _clean_np(chunk)
        i = chunk.end
        if i + 1 < len(doc) and doc[i].text.lower() == "of":
            of_chunk = _np_after(doc, i + 1)
            if of_chunk is not None:
                c.object_of = _clean_np(of_chunk)[0]
                c.object = f"{c.object} of {c.object_of}"
                i = of_chunk.end
    if i < len(doc) and doc[i].text.lower() in RESULT_WORDS:
        c.stop = c.stop or doc[i].text.lower()
        i += 1

    while i < len(doc):
        prep, n = _match_prep(doc, i)
        if not prep:
            i += 1
            continue
        if prep == "out of" and i + n < len(doc) and doc[i + n].text.lower() == "view":
            c.direction = c.direction or "out of view"
            i += n + 1
            continue
        if prep == "from":
            # "from beside the sink": the second preposition names the place.
            prep2, n2 = _match_prep(doc, i + n)
            if prep2:
                n += n2
        target = _np_after(doc, i + n)
        if target is None:
            i += n
            continue
        phrase = _clean_np(target)[0]
        if prep in L.PLACE_FROM_PREPS and not c.place_from:
            c.place_from = phrase
        elif prep in L.LANDMARK_PREPS and not c.object:
            c.landmarks.append(phrase)
            # "past the island and the fridge": keep each landmark.
            while target.end + 1 < len(doc) and doc[target.end].text.lower() == "and":
                more = _np_after(doc, target.end + 1)
                if more is None:
                    break
                c.landmarks.append(_clean_np(more)[0])
                target = more
        elif prep in L.PLACE_TO_PREPS and not c.place:
            c.place_prep, c.place = prep, phrase
        i = target.end

    object_turn = c.verb == "turn" and (c.particle in L.OBJECT_TURN_PARTICLES or bool(c.object))
    c.is_navigation = c.verb in L.NAV_VERBS and not c.object and not object_turn
    if not c.object and not c.is_navigation and c.landmarks:
        # "Reach toward the pillow": the target of a reach is its object.
        c.object, c.object_head = c.landmarks[0], c.landmarks[0].split()[-1]
        c.landmarks = []
    if c.is_navigation and c.place and not c.landmarks:
        c.landmarks.append(c.place)
        c.place = c.place_prep = ""
    if not c.object and not c.is_navigation and c.verb not in L.FAILURE_WORDS:
        warn.append("no object")
    if c.object_head == "hands" and c.determiner == "your" and not c.actor:
        c.actor = "both hands"
    return c, warn


def _counts(caption: str) -> Tuple[int, int, int, List[str]]:
    words = len(caption.split())
    commas = caption.count(",")
    inner_ends = len(re.findall(r"[.!?](?=\s+[A-Za-z])", caption))
    conns = [m.group(1).lower() for m in CONNECTOR_RE.finditer(caption)]
    return words, commas, inner_ends, conns


@lru_cache(maxsize=None)
def parse_caption(caption: str) -> ParsedCaption:
    """Rules plus spaCy. Pure and cached, so it is safe to call many times."""
    raw = str(caption)
    words, commas, ends, conns = _counts(raw)
    norm = L.normalize_literal(raw)
    exact = raw.strip() in L.FIXED_LITERALS
    if exact or norm in L.FIXED_LITERALS:
        return ParsedCaption(raw, True, norm, [], "literal",
                             [] if exact else ["literal not exact"], words, commas, ends, conns)
    for lit in L.FIXED_LITERALS:
        if re.search(r"\b" + lit + r"\b", raw):
            # A literal mixed with other words. Parse the rest, but keep the literal.
            raw_rest = re.sub(r"^(while|and|then|but)\s+", "", raw.replace(lit, "").strip(), flags=re.I)
            pc = parse_caption(raw_rest) if raw_rest else None
            clauses = pc.clauses if pc else []
            return ParsedCaption(raw, False, lit, clauses, "rules_partial",
                                 ["literal mixed with other words"], words, commas, ends, conns)

    clauses, warnings = [], []
    text = re.sub(r"[.!?]+$", "", raw.strip())
    parts: List[Tuple[str, str]] = []
    for s_i, sentence in enumerate(re.split(r"[.!?]+\s+", text)):
        sub = _split(sentence)
        if s_i > 0 and sub:
            sub[0] = (".", sub[0][1])
        parts += sub
    for k, (conn, part) in enumerate(parts):
        cl, w = _parse_clause(k, conn, part)
        clauses.append(cl)
        warnings += [f"clause {k}: {x}" for x in w]

    # Attach held objects to the clause they qualify, and mark shared actors.
    for k, cl in enumerate(clauses):
        if cl.is_holding and k > 0:
            clauses[k - 1].held_object, clauses[k - 1].held_actor = cl.held_object, cl.held_actor
    actors = [cl.actor for cl in clauses if cl.actor and not cl.is_holding]
    for k, cl in enumerate(clauses):
        if not cl.actor and not cl.is_holding and not cl.is_navigation and actors:
            later_and = any(c2.connector == "and" and c2.actor for c2 in clauses[k + 1:])
            if later_and:
                cl.actor_shared = True
    source = "rules" if not warnings else "rules_partial"
    return ParsedCaption(raw, False, "", clauses, source, warnings, words, commas, ends, conns)


# Sol fallback ----------------------------------------------------------------

_CLAUSE_FIELDS = {
    "connector": "string", "adverb": "string", "intent": "string", "verb": "string",
    "particle": "string", "object": "string", "object_head": "string", "attributes": "array",
    "actor": "string", "tool": "string", "place_prep": "string", "place": "string",
    "place_from": "string", "landmarks": "array", "direction": "string", "facing": "string",
    "stop": "string", "held_object": "string", "held_actor": "string",
    "is_navigation": "boolean", "is_holding": "boolean",
}

SOL_PARSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["clauses"],
    "properties": {
        "clauses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(_CLAUSE_FIELDS),
                "properties": {
                    k: ({"type": "array", "items": {"type": "string"}} if v == "array" else {"type": v})
                    for k, v in _CLAUSE_FIELDS.items()
                },
            },
        }
    },
}

SOL_PARSE_PROMPT = """Split this robot instruction caption into action clauses and extract fields.
Use lower case. Leave a field empty when the caption does not state it. Do not invent.
verb is the base form. particle is a phrasal particle such as "up" in "pick up".
actor is "right hand", "left hand", "both hands" or a body part such as "hip".
place is a destination or location. place_from is a source. landmarks are places the ego walks
toward or past. is_navigation is true when the ego itself moves (walk, turn, step).
A "while holding X in your Y hand" clause has is_holding true.
Caption: {caption}"""


def parse_with_sol(caption: str, sol) -> Optional[ParsedCaption]:
    """Use Sol when the rules could not finish. Returns None when Sol is unavailable."""
    if sol is None or not sol.available:
        return None
    data = sol.json(SOL_PARSE_PROMPT.format(caption=caption), SOL_PARSE_SCHEMA, name="caption_fields")
    if not data:
        return None
    words, commas, ends, conns = _counts(caption)
    clauses = []
    for k, d in enumerate(data.get("clauses", [])):
        cl = Clause(index=k, text="", **{f: d.get(f, [] if t == "array" else ("" if t == "string" else False))
                                         for f, t in _CLAUSE_FIELDS.items()})
        cl.verb_key = f"{cl.verb} {cl.particle}".strip()
        cl.has_failure_word = cl.verb in L.FAILURE_WORDS
        clauses.append(cl)
    return ParsedCaption(caption, False, "", clauses, "sol", [], words, commas, ends, conns)


def parse(caption: str, sol=None) -> ParsedCaption:
    """Rules first. Sol only when the rules leave a warning."""
    pc = parse_caption(caption)
    if pc.source == "rules_partial" and sol is not None:
        alt = parse_with_sol(caption, sol)
        if alt is not None:
            alt.warnings = ["rules: " + w for w in pc.warnings]
            return alt
    return pc
