"""Load recordings and segments, and fill the platform fields the samples lack.

The synthetic values are seeded, so every run gives the same data. A few planted cases
make the platform checks show something on the samples. Every synthetic value is marked.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import lexicon as L
from .schema import PLATFORM_COLUMNS, SEGMENT_COLUMNS

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi")


# Videos ---------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_videos(data_dir: Path, drive_folder: str, log=print) -> pd.DataFrame:
    """Make sure every video listed in videos.sha256 is present and unchanged.

    A missing video is downloaded from the public Google Drive folder. A video whose checksum
    differs from the manifest stops the run: the checks must see the frames the annotators saw.
    """
    data_dir = Path(data_dir)
    manifest = data_dir / "videos.sha256"
    if not manifest.exists():
        log("No videos.sha256 manifest. Using the videos as they are.")
        return pd.DataFrame(columns=["file", "status"])
    expected = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            expected[name.strip()] = digest
    missing = [n for n in expected if not (data_dir / n).exists()]
    if missing:
        import gdown

        log(f"Downloading {len(missing)} video(s) from Google Drive: {', '.join(missing)}")
        listing = {f.path: f.id for f in gdown.download_folder(drive_folder, skip_download=True, quiet=True)}
        for name in missing:
            if name not in listing:
                raise FileNotFoundError(f"{name} is not in the Drive folder {drive_folder}")
            gdown.download(id=listing[name], output=str(data_dir / name), quiet=False)
    rows = []
    for name, digest in expected.items():
        ok = _sha256(data_dir / name) == digest
        rows.append({"file": name, "status": "downloaded" if name in missing else "present",
                     "checksum": "match" if ok else "MISMATCH"})
        if not ok:
            raise ValueError(f"{name} does not match videos.sha256. Replace it with the original file.")
    return pd.DataFrame(rows)


# Loading -------------------------------------------------------------------

def discover_recordings(data_dir: Path) -> pd.DataFrame:
    """Use recordings.csv when present. Otherwise pair <Name>.mp4 with <name>_annotations.csv."""
    data_dir = Path(data_dir)
    manifest = data_dir / "recordings.csv"
    if manifest.exists():
        rec = pd.read_csv(manifest, dtype=str)
        rec["path"] = [str((data_dir / p) if not Path(p).is_absolute() else p) for p in rec["path"]]
        return rec[["recording_id", "path"]]
    rows = []
    for video in sorted(data_dir.iterdir()):
        if video.suffix.lower() in VIDEO_SUFFIXES:
            rows.append({"recording_id": video.stem.lower(), "path": str(video)})
    if not rows:
        raise FileNotFoundError(f"No videos found in {data_dir}")
    return pd.DataFrame(rows)


def load_segments(recordings: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Read segments.csv when present. Otherwise read <recording_id>_annotations.csv per video."""
    data_dir = Path(data_dir)
    full = data_dir / "segments.csv"
    if full.exists():
        seg = pd.read_csv(full)
    else:
        frames = []
        by_lower = {p.name.lower(): p for p in data_dir.iterdir()}
        for rid in recordings["recording_id"]:
            path = by_lower.get(f"{rid}_annotations.csv")
            if path is None:
                raise FileNotFoundError(f"No annotations CSV for {rid}")
            df = pd.read_csv(path)
            df = df.rename(columns={"#": "segment_index"})
            df.insert(0, "recording_id", rid)
            frames.append(df)
        seg = pd.concat(frames, ignore_index=True)
    if "segment_index" not in seg:
        seg["segment_index"] = seg.groupby("recording_id").cumcount() + 1
    if "segment_id" not in seg:
        seg["segment_id"] = [f"{r}_{int(i):03d}" for r, i in zip(seg.recording_id, seg.segment_index)]
    for col in PLATFORM_COLUMNS:
        if col not in seg:
            seg[col] = np.nan
    seg["start_s"] = seg["start_s"].astype(float)
    seg["end_s"] = seg["end_s"].astype(float)
    seg["caption"] = seg["caption"].astype(str)
    ordered = [c for c in SEGMENT_COLUMNS if c in seg] + [c for c in seg if c not in SEGMENT_COLUMNS]
    return seg[ordered].sort_values(["recording_id", "segment_index"]).reset_index(drop=True)


def validate_segments(seg: pd.DataFrame) -> pd.DataFrame:
    """Contract checks on the input itself. Returns one row per problem."""
    problems = []

    def add(row, column, issue):
        problems.append({"segment_id": row.get("segment_id", ""), "column": column, "issue": issue})

    missing = [c for c in SEGMENT_COLUMNS if c not in seg]
    for c in missing:
        problems.append({"segment_id": "*", "column": c, "issue": "column missing"})
    for _, row in seg.iterrows():
        if pd.isna(row["caption"]) or not str(row["caption"]).strip():
            add(row, "caption", "blank caption")
        if not np.isfinite(row["start_s"]) or not np.isfinite(row["end_s"]):
            add(row, "start_s/end_s", "time is not a finite number")
        if isinstance(row.get("accepted_type"), str) and row["accepted_type"] not in L.ACCEPTED_TYPES:
            add(row, "accepted_type", f"unknown value {row['accepted_type']!r}")
        sugg = row.get("suggestions")
        if isinstance(sugg, str):
            try:
                items = json.loads(sugg)
                if not isinstance(items, list) or len(items) != 3:
                    add(row, "suggestions", "not a list of 3")
            except json.JSONDecodeError:
                add(row, "suggestions", "not valid JSON")
        for col in ("caption_created_at", "caption_updated_at"):
            v = row.get(col)
            if isinstance(v, str) and pd.isna(pd.to_datetime(v, utc=True, errors="coerce")):
                add(row, col, "not a timestamp")
    if seg["segment_id"].duplicated().any():
        for sid in seg.loc[seg["segment_id"].duplicated(), "segment_id"]:
            problems.append({"segment_id": sid, "column": "segment_id", "issue": "duplicate id"})
    return pd.DataFrame(problems, columns=["segment_id", "column", "issue"])


# Synthetic platform fields ---------------------------------------------------

VERB_SWAPS = {"Pick up": "Grab", "Put": "Place", "Place": "Put", "Wash": "Clean", "Rinse": "Wash",
              "Move": "Slide", "Walk": "Go", "Turn on": "Switch on", "Turn off": "Switch off",
              "Open": "Pull open", "Seal": "Close", "Flatten": "Press", "Drop": "Place"}
LITERAL_SUGGESTIONS = ["IDLE", "Look around the room", "Stand still"]

# Planted cases for the demo: (segment_id, field, what is planted).
DEMO_PLANTS = [
    ("sample1_009", "caption_updated_at", "edited after segments 10 to 12 were created (backward edit)"),
    ("sample2_010", "accepted_type", "says accepted_without_edits but the caption matches no suggestion"),
    ("sample3_072..089", "caption_created_at", "18 captions created 2 to 4 s apart (copied block)"),
    ("sample3_046..090", "annotator_id", "second annotator takes over the recording"),
    ("failure captions", "suggestions", "closest suggestion is the clean action without the failure"),
]


def _swap_hand(c: str) -> str:
    if "right hand" in c:
        return c.replace("right hand", "left hand")
    if "left hand" in c:
        return c.replace("left hand", "right hand")
    if "both hands" in c:
        return c.replace("both hands", "your right hand").replace("with your your", "with your")
    return c + " with your right hand"


def _drop_hand(c: str) -> str:
    out = L.HAND_RE.sub("", c)
    return re.sub(r"\s{2,}", " ", out).strip()


def _swap_verb(c: str) -> str:
    for a, b in VERB_SWAPS.items():
        if c.startswith(a + " "):
            return b + c[len(a):]
    return c


def _clean_failure(c: str) -> str:
    out = re.sub(r"^(Attempt|Try) to (\w)", lambda m: m.group(2).upper(), c)
    out = re.split(r"\s+but\s+", out)[0]
    return out.strip()


def _is_failure(c: str) -> bool:
    words = set(re.findall(r"[a-z]+", c.lower()))
    return bool(L.INTENT_RE.search(c)) or bool(words & L.FAILURE_WORDS)


def _make_suggestions(caption: str, rng: np.random.Generator, exact: bool) -> List[str]:
    if caption.strip() in L.FIXED_LITERALS:
        out = list(LITERAL_SUGGESTIONS)
        if caption.strip() not in out:
            out[0] = caption.strip()
        rng.shuffle(out)
        return out
    if _is_failure(caption):
        # Suggestions describe the clean action, so none of them records the failure.
        close = _clean_failure(caption)
        pool = [close, _swap_hand(close), _swap_verb(close), _drop_hand(close)]
    else:
        if exact:
            close = caption
        else:
            close = _drop_hand(caption) if L.HAND_RE.search(caption) else _swap_verb(caption)
        pool = [close, _swap_hand(caption), _swap_verb(caption), _drop_hand(caption), _swap_hand(close)]
    out: List[str] = []
    for s in pool:
        if s and s not in out and (s != caption or s == close):
            out.append(s)
        if len(out) == 3:
            break
    while len(out) < 3:
        out.append(f"{close} slowly" if len(out) == 1 else "Walk forward")
    rng.shuffle(out)
    return out


def add_synthetic_platform_fields(seg: pd.DataFrame, seed: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing platform fields with seeded values. Existing values are kept."""
    rng = np.random.default_rng(seed)
    seg = seg.copy()
    filled = [c for c in PLATFORM_COLUMNS if seg[c].isna().all()]
    if not filled:
        return seg, pd.DataFrame(columns=["segment_id", "field", "plant"])
    base = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)
    annotators = [f"ann_{i:02d}" for i in range(1, 9)]
    created, updated, ann, sugg, acc = {}, {}, {}, {}, {}

    for r_i, (rid, grp) in enumerate(seg.groupby("recording_id", sort=True)):
        grp = grp.sort_values("segment_index")
        first, second = annotators[(2 * r_i) % 8], annotators[(2 * r_i + 1) % 8]
        split_at = int(grp.segment_index.max() * 0.5) + 1 if len(grp) > 60 else None
        clock = base + timedelta(hours=3 * r_i)
        for _, row in grp.iterrows():
            idx, sid, cap = int(row.segment_index), row.segment_id, row.caption
            ann[sid] = second if split_at and idx >= split_at else first
            exact = bool(rng.random() < 0.5)
            s = _make_suggestions(cap, rng, exact)
            sugg[sid] = json.dumps(s)
            acc[sid] = "accepted_without_edits" if cap in s else "accepted_with_edits"
            dur = float(row.end_s - row.start_s)
            work = max(6.0, dur * rng.uniform(2.0, 4.0)) + (0 if cap in s else rng.uniform(8, 25))
            clock = clock + timedelta(seconds=float(work))
            created[sid] = clock
            updated[sid] = clock

    # Planted cases (only where the ids exist).
    ids = set(seg.segment_id)
    if {"sample1_009", "sample1_012"} <= ids:
        updated["sample1_009"] = created["sample1_012"] + timedelta(seconds=40)
    if "sample2_010" in ids:
        acc["sample2_010"] = "accepted_without_edits"
        s = json.loads(sugg["sample2_010"])
        sugg["sample2_010"] = json.dumps([x for x in s if x != seg.set_index("segment_id").caption["sample2_010"]][:2] + ["Open the bag"])
    copy_block = [f"sample3_{i:03d}" for i in range(72, 90)]
    if set(copy_block) <= ids:
        t = created[copy_block[0]]
        for sid in copy_block:
            t = t + timedelta(seconds=float(rng.uniform(2, 4)))
            created[sid] = updated[sid] = t
        # Later segments start after the pasted block.
        later = [f"sample3_{i:03d}" for i in range(90, 91) if f"sample3_{i:03d}" in ids]
        for sid in later:
            t = t + timedelta(seconds=20)
            created[sid] = updated[sid] = t

    iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")
    values = {
        "annotator_id": seg["segment_id"].map(ann),
        "suggestions": seg["segment_id"].map(sugg),
        "accepted_type": seg["segment_id"].map(acc),
        "caption_created_at": seg["segment_id"].map(lambda s: iso(created[s])),
        "caption_updated_at": seg["segment_id"].map(lambda s: iso(updated[s])),
    }
    for col in filled:
        seg[col] = values[col]
    seg["synthetic_fields"] = ",".join(filled)
    plants = pd.DataFrame(DEMO_PLANTS, columns=["segment_id", "field", "plant"])
    return seg, plants
