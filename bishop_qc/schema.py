"""Shared types: the data contract and the records that primitives and checks pass around."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Data contract -------------------------------------------------------------

RECORDING_COLUMNS = {
    "recording_id": "Unique id of the recording.",
    "path": "Path to the video file.",
}

SEGMENT_COLUMNS = {
    "recording_id": "Recording the segment belongs to.",
    "segment_id": "Unique id of the segment, <recording_id>_<index>.",
    "segment_index": "Position of the segment in the recording, from 1.",
    "start_s": "Start time in seconds.",
    "end_s": "End time in seconds.",
    "caption": "Final caption text.",
    "annotator_id": "Annotator who wrote the caption.",
    "suggestions": "JSON list of the 3 suggestions the annotator saw.",
    "accepted_type": "accepted_without_edits or accepted_with_edits.",
    "caption_created_at": "UTC time the caption was first committed.",
    "caption_updated_at": "UTC time of the last edit to the caption.",
}

# Fields the sample CSVs do not have. The loader fills them when synthesis is on.
PLATFORM_COLUMNS = ["annotator_id", "suggestions", "accepted_type", "caption_created_at",
                    "caption_updated_at"]


# Parser output -------------------------------------------------------------

@dataclass
class Clause:
    """One action clause of a caption."""

    index: int
    text: str
    connector: str = ""        # word that joins this clause to the one before it
    adverb: str = ""           # e.g., "Repeatedly"
    intent: str = ""           # e.g., "attempt to"
    verb: str = ""             # lemma, e.g., "pick"
    particle: str = ""         # e.g., "up"
    verb_key: str = ""         # verb with particle, e.g., "pick up"
    object: str = ""           # noun phrase without the determiner
    object_head: str = ""      # head noun of the object
    object_of: str = ""        # "of" phrase, e.g., "black parts" in "bag of black parts"
    determiner: str = ""       # "the", "a", "your"
    attributes: List[str] = field(default_factory=list)
    actor: str = ""            # "right hand", "left hand", "both hands", or a body part
    actor_shared: bool = False # actor taken from a sibling clause in the same sentence
    tool: str = ""             # e.g., "fork" in "with the fork in your right hand"
    place_prep: str = ""
    place: str = ""            # destination or location
    place_from: str = ""       # source
    landmarks: List[str] = field(default_factory=list)
    direction: str = ""        # "left", "right", "upside down"
    facing: str = ""           # target after "until you are facing"
    stop: str = ""             # "until ..." text
    held_object: str = ""
    held_actor: str = ""
    is_navigation: bool = False
    is_holding: bool = False
    has_failure_word: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedCaption:
    caption: str
    is_fixed_literal: bool
    literal: str                     # the matching literal, or ""
    clauses: List[Clause]
    source: str                      # "rules", "rules_partial", "sol", "literal"
    warnings: List[str] = field(default_factory=list)
    word_count: int = 0
    comma_count: int = 0
    sentence_ends: int = 0
    connectors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["clauses"] = [c.to_dict() for c in self.clauses]
        return d


# Model answers and findings ------------------------------------------------

@dataclass
class Answer:
    """One answer from an API model. A failure is unassessable, never a pass."""

    question: str
    status: str                       # "ok" or "unassessable"
    source: str                       # "openai_decisions", "sol", "sol_fallback", "jev", "none"
    value: Any = None                 # True/False, an option, or a level
    prob: Optional[float] = None      # probability of value
    probs: Optional[Dict[str, float]] = None
    error: str = ""


@dataclass
class Finding:
    """One result of one check on one segment or boundary."""

    row: str                          # row(s) of the issue sheet, e.g., "2" or "2, 3"
    check: str                        # check key, e.g., "row02_03"
    recording_id: str
    segment_id: str                   # for a boundary: the segment that starts at it
    status: str                       # see FINDING_STATUSES
    message: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


FINDING_STATUSES = {
    "candidate": "A possible error. Goes to a human reviewer. Not a verdict.",
    "pass": "Checked, and no sign of the error.",
    "unassessable": "A signal or a model answer was missing. Never counted as a pass.",
    "report": "A measurement for the recording, not a per segment verdict.",
    "no_coverage": "The inputs cannot show this error at all.",
}
