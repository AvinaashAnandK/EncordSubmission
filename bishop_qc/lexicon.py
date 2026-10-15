"""Literals, word lists, regex and mappings shared by the parser and the checks.

Every list here is a starting point. Grow it from review results each week.
Lists marked CLIENT need sign off before a check that uses them counts in a score.
"""
from __future__ import annotations

import re

# Guideline literals ---------------------------------------------------------

FIXED_LITERALS = ("IDLE", "SITTING", "TASK UNRELATED ACTION")
FALLBACK_NOUNS = ("garment", "container", "utensil", "linen")
ACCEPTED_TYPES = ("accepted_without_edits", "accepted_with_edits")

# Actors ---------------------------------------------------------------------

HAND_RE = re.compile(
    r"\b(?:with|using|in)\s+(?:your\s+|the\s+|both\s+)?(?P<side>left|right|both)\s+hands?\b"
    r"|\b(?:with|using)\s+(?P<both>both)\s+hands\b",
    re.I,
)
BODY_PARTS = ("hip", "foot", "feet", "knee", "elbow", "shoulder", "forearm", "leg", "head", "chin")
BODY_RE = re.compile(r"\b(?:with|using)\s+your\s+(?P<part>" + "|".join(BODY_PARTS) + r")\b", re.I)
TOOL_RE = re.compile(
    r"\bwith\s+the\s+(?P<tool>[a-z ]+?)\s+in\s+your\s+(?P<side>left|right)\s+hand\b", re.I
)
HOLDING_RE = re.compile(
    r"\bholding\s+(?P<obj>.+?)\s+(?:in|with)\s+(?:your\s+)?(?P<side>left|right|both)\s+hands?\b", re.I
)

# Verbs ----------------------------------------------------------------------

NAV_VERBS = {"walk", "step", "turn", "go", "head", "move", "approach", "return", "enter", "exit",
             "leave", "climb", "descend", "proceed", "back"}
# A turn with one of these particles acts on an object, e.g., "Turn on the tap".
OBJECT_TURN_PARTICLES = {"on", "off", "over", "up", "down"}
PHRASAL_VERBS = {
    "pick up", "put down", "turn on", "turn off", "switch on", "switch off", "take out", "take off",
    "put away", "set down", "spread out", "shake out", "lift up", "put on", "wipe down", "rinse off",
    "turn over", "hand over", "throw away", "put back", "pull out", "push down", "clean up",
    "fill up", "open up", "line up", "pull up", "pick out", "lay out", "wipe off", "dry off",
}
PHRASAL_PARTICLES = {"up", "down", "on", "off", "out", "over", "away", "aside", "back", "apart", "together"}
ADJUSTMENT_VERBS = {"adjust", "regrip", "reposition", "shift", "settle", "readjust", "tighten", "loosen"}
# CLIENT: actions with a defined end point (row 36). Placeholder until signed off.
END_STATE_VERBS = {"close", "open", "twist", "screw", "unscrew", "fill", "empty", "fold", "unfold",
                   "zip", "unzip", "seal", "tie", "lock", "unlock", "plug", "unplug", "cover",
                   "uncover", "turn on", "turn off", "switch on", "switch off"}
OPPOSITE_VERBS = {
    frozenset(p) for p in [
        ("throw", "catch"), ("open", "close"), ("pick up", "put down"), ("pick up", "place"),
        ("turn on", "turn off"), ("push", "pull"), ("lift", "lower"), ("insert", "remove"),
        ("plug", "unplug"), ("fold", "unfold"), ("zip", "unzip"), ("lock", "unlock"),
        ("fill", "empty"), ("put", "take"), ("switch on", "switch off"), ("seal", "open"),
    ]
}
# House preferences (advice only, row 83).
HOUSE_VERB_MAP = {"grab": "pick up", "take": "pick up", "set": "place", "lay": "place",
                  "go": "walk", "head": "walk", "proceed": "walk", "stroll": "walk"}

# Words and phrases ------------------------------------------------------------

INTENT_PHRASES = ("attempt to", "attempts to", "try to", "tries to", "trying to", "want to",
                  "wants to", "intend to", "intends to", "aim to", "plan to", "start to",
                  "starts to", "begin to", "begins to")
INTENT_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in INTENT_PHRASES) + r")\b", re.I)
FAILURE_WORDS = {"miss", "misses", "missed", "fail", "fails", "failed", "drop", "drops", "dropped",
                 "slip", "slips", "slipped", "retract", "retracts", "withdraw", "withdraws",
                 "fumble", "fumbles", "spill", "spills", "knock", "knocks", "lose", "loses"}
CONTEXT_WORDS = ("it", "its", "them", "they", "another", "other", "same", "again", "back",
                 "previous", "last", "remaining", "rest", "also")
CONTEXT_RE = re.compile(r"\b(" + "|".join(CONTEXT_WORDS) + r")\b", re.I)
GRIP_WORDS = ("finger", "fingers", "thumb", "thumbs", "grip", "grips", "gripping", "palm", "palms",
              "wrist", "fingertip", "fingertips", "knuckle", "knuckles", "tightly", "firmly",
              "gently", "carefully", "loosely")
GRIP_RE = re.compile(r"\b(" + "|".join(GRIP_WORDS) + r")\b", re.I)
VAGUE_NOUNS = {"thing", "things", "item", "items", "object", "objects", "stuff", "something",
               "part", "parts", "piece", "pieces"}
FALLBACK_SYNONYMS = {"fabric": "garment", "clothing": "garment", "clothes": "garment",
                     "apparel": "garment", "vessel": "container", "receptacle": "container",
                     "implement": "utensil", "cutlery": "utensil", "textile": "linen",
                     "bedding": "linen"}
CONFUSABLE_PAIRS = [("bowl", "plate"), ("cup", "glass"), ("cup", "mug"), ("towel", "cloth"),
                    ("towel", "blanket"), ("jar", "bottle"), ("can", "bottle"), ("pan", "pot"),
                    ("spoon", "ladle"), ("knife", "spatula"), ("box", "bag"), ("lid", "plate")]
CONTAINER_NOUNS = {"bag", "jar", "bottle", "box", "bowl", "cup", "glass", "can", "container",
                   "packet", "pack", "carton", "tin", "tub", "pot", "pan", "basket", "mug"}
CONTENTS_RE = re.compile(r"\b(full of|filled with|containing|with some)\b", re.I)

# Places and navigation -------------------------------------------------------

DIRECTION_WORDS = {"left", "right", "forward", "forwards", "back", "backward", "backwards",
                   "ahead", "straight", "around", "sideways"}
LANDMARK_PREPS = ("toward", "towards", "past", "to", "through", "around", "into", "across",
                  "along", "between", "beside", "by")
# Multiword preposition first, so the longest match wins.
PLACE_TO_PREPS = ("on top of", "in front of", "next to", "onto", "into", "inside", "on", "in",
                  "under", "over", "beside", "behind", "to", "above", "below", "near")
PLACE_FROM_PREPS = ("out of", "from", "off")
FACING_RE = re.compile(r"\buntil\s+you\s+are\s+facing\s+(?:the\s+)?(?P<target>[a-z ]+)", re.I)
DIRECTION_RE = re.compile(r"\b(?:to the (?P<side>left|right)|(?P<updown>upside down))\b", re.I)

STATIC_FIXTURES = {
    "wall", "doorway", "door", "window", "counter", "countertop", "kitchen counter",
    "kitchen island", "island", "sink", "staircase", "stairs", "stove", "oven", "fridge",
    "refrigerator", "cabinet", "cupboard", "shelf", "shelves", "sofa", "couch", "table",
    "coffee table", "desk", "bed", "dishwasher", "washing machine", "bathtub", "shower",
    "toilet", "fireplace", "hallway", "corridor", "entrance", "closet", "wardrobe", "bookshelf",
    "bookcase", "dresser", "tap", "faucet", "kitchen", "living room", "bedroom", "bathroom",
}
# CLIENT: furniture that may or may not count as static (row 53).
GREY_FURNITURE = {"chair", "folding chair", "stool", "ottoman", "bench", "cart", "trolley",
                  "laundry basket", "basket"}
PERSON_WORDS = {"person", "man", "woman", "child", "people", "dog", "cat"}

# Language ------------------------------------------------------------------

CONNECTORS_ALLOWED = ("then", "while", "and", "but", "until")
CONNECTORS_OTHER = ("after", "before", "as", "when", "so", "because", "once", "since",
                    "whilst", "meanwhile", "afterwards", "later", "next")
# CLIENT: the guideline sets no dialect. Report only (row 71).
DIALECT_PAIRS = {"tap": "faucet", "hob": "stovetop", "bin": "trash can", "cupboard": "cabinet",
                 "worktop": "countertop", "tea towel": "dish towel", "torch": "flashlight",
                 "trolley": "cart", "kitchen roll": "paper towel", "cooker": "stove",
                 "jug": "pitcher", "washing-up liquid": "dish soap"}
UK_US_SPELLINGS = {"colour": "color", "colours": "colors", "grey": "gray", "centre": "center",
                   "metre": "meter", "litre": "liter", "aluminium": "aluminum",
                   "favourite": "favorite", "mould": "mold", "tyre": "tire", "yoghurt": "yogurt",
                   "organise": "organize", "organised": "organized", "moulded": "molded"}


def normalize_literal(caption: str) -> str:
    """Case and separator variants only (row 81). No fuzzy matching."""
    return re.sub(r"[\s_\-]+", " ", caption.strip().rstrip(".!?")).upper()
