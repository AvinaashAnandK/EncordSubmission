"""P8 suggestion features and P9 edit time features, from the platform fields."""
from __future__ import annotations

import json
from typing import Callable, List

import numpy as np
import pandas as pd

from . import lexicon as L
from .schema import ParsedCaption


def _norm_dist(a: str, b: str) -> float:
    try:
        from rapidfuzz.distance import Levenshtein

        return float(Levenshtein.normalized_distance(a, b))
    except ImportError:
        import difflib

        return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def _main_clause(pc: ParsedCaption):
    return next((c for c in pc.clauses if not c.is_holding), None)


def _has_failure(pc: ParsedCaption, text: str) -> bool:
    return bool(L.INTENT_RE.search(text)) or any(c.has_failure_word for c in pc.clauses)


def suggestion_features(seg: pd.DataFrame, parse: Callable[[str], ParsedCaption]) -> pd.DataFrame:
    """P8. Distance from the caption to the nearest suggestion, and which fields changed."""
    rows = []
    for r in seg.itertuples():
        base = {"segment_id": r.segment_id}
        if not isinstance(r.suggestions, str):
            rows.append({**base, "p8_status": "no_data"})
            continue
        sugg: List[str] = json.loads(r.suggestions)
        dists = [_norm_dist(r.caption, s) for s in sugg]
        k = int(np.argmin(dists))
        near = sugg[k]
        pc, ps = parse(r.caption), parse(near)
        mc, ms = _main_clause(pc), _main_clause(ps)
        same = lambda f: (getattr(mc, f) if mc else "") == (getattr(ms, f) if ms else "")
        exact = r.caption in sugg
        rows.append({
            **base, "p8_status": "ok",
            "p8_nearest_idx": k, "p8_nearest_text": near, "p8_nearest_dist": round(dists[k], 4),
            "p8_exact_match": exact,
            "p8_accepted_type_consistent": (r.accepted_type == "accepted_without_edits") == exact,
            "p8_verb_changed": not same("verb_key"),
            "p8_object_changed": not same("object"),
            "p8_actor_changed": not same("actor"),
            "p8_place_changed": not same("place"),
            "p8_failure_added": _has_failure(pc, r.caption) and not _has_failure(ps, near),
        })
    return pd.DataFrame(rows)


def _repeat_blocks(caps: List[str], durs: List[float], min_run: int) -> List[int]:
    """For each segment, the index of the earlier segment it repeats, or -1.

    A repeat is a run of at least min_run segments whose captions and durations (to 0.1 s)
    equal an earlier run.
    """
    n = len(caps)
    repeat_of = [-1] * n
    for d in range(1, n):
        k = 0
        while k + d < n:
            j = k
            while j + d < n and caps[j] == caps[j + d] and abs(durs[j] - durs[j + d]) < 0.05:
                j += 1
            if j - k >= min_run:
                for m in range(k, j):
                    if repeat_of[m + d] == -1:
                        repeat_of[m + d] = m
            k = max(j, k + 1)
    return repeat_of


def edit_time_features(seg: pd.DataFrame, min_repeat_run: int = 5) -> pd.DataFrame:
    """P9. Time per caption, backward edits, annotator switches and repeated blocks."""
    out = []
    for rid, grp in seg.sort_values(["recording_id", "segment_index"]).groupby("recording_id"):
        grp = grp.copy()
        created = pd.to_datetime(grp.caption_created_at, utc=True, errors="coerce")
        updated = pd.to_datetime(grp.caption_updated_at, utc=True, errors="coerce")
        has_time = created.notna().all()
        f = pd.DataFrame({"segment_id": grp.segment_id})
        if has_time:
            f["p9_edit_lag_s"] = (updated - created).dt.total_seconds().to_numpy()
            f["p9_gap_prev_s"] = created.diff().dt.total_seconds().to_numpy()
            later_min = created[::-1].cummin()[::-1].shift(-1)
            f["p9_backward_edit"] = ((updated > later_min) & (updated > created)).fillna(False).to_numpy()
            c_arr = created.to_numpy()
            f["p9_later_created_before_update"] = [
                int((c_arr[i + 1:] < u).sum()) if u > c else 0
                for i, (c, u) in enumerate(zip(c_arr, updated.to_numpy()))]
        else:
            for col in ("p9_edit_lag_s", "p9_gap_prev_s", "p9_backward_edit", "p9_later_created_before_update"):
                f[col] = np.nan
        f["p9_annotator_switch"] = (grp.annotator_id != grp.annotator_id.shift()).to_numpy() & \
            grp.annotator_id.shift().notna().to_numpy()
        caps = grp.caption.tolist()
        durs = (grp.end_s - grp.start_s).round(1).tolist()
        rep = _repeat_blocks(caps, durs, min_repeat_run)
        ids = grp.segment_id.tolist()
        f["p9_repeat_of"] = [ids[k] if k >= 0 else "" for k in rep]
        out.append(f)
    return pd.concat(out, ignore_index=True)
