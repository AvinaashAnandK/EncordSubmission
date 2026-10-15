"""Run checks on a made-up list of captions, with no video, and compare what they flag with what is expected.

Use it for categories the sample data does not cover, e.g., navigation. The captions form one test recording,
in order, so checks that read the next segment see it. Frame based parts of a check find no frames and report
unassessable, which is expected.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Sequence

import pandas as pd

from .. import parser as P
from .runner import CheckContext, CheckResult, run_check

FRAME_COLUMNS = ["recording_id", "frame_idx", "t", "is_timeline", "near_boundary", "path", "blur", "segment_id"]


def caption_context(cfg, clients, cases: Sequence[dict], recording_id: str = "test") -> CheckContext:
    """cases: dicts with "caption" (and "expect", "why" for the comparison)."""
    seg = pd.DataFrame([{
        "recording_id": recording_id, "segment_index": i + 1, "segment_id": f"{recording_id}_{i + 1:03d}",
        "start_s": 3.0 * i, "end_s": 3.0 * (i + 1), "caption": c["caption"],
    } for i, c in enumerate(cases)])
    prims = SimpleNamespace(
        segments=seg,
        parsed={r.segment_id: P.parse_caption(r.caption) for r in seg.itertuples()},
        frames=pd.DataFrame(columns=FRAME_COLUMNS),
        hand_timeline=pd.DataFrame(), hand_dets=pd.DataFrame(), motion=pd.DataFrame(), objects=pd.DataFrame(),
        recordings=pd.DataFrame([{"recording_id": recording_id, "p1_duration_s": 3.0 * len(seg), "p1_fps_measured": 30.0}]),
        cost_per_segment=pd.DataFrame(columns=["name", "segment_id", "seconds", "seconds_uncached", "total_usd"]),
        thresholds={},
    )
    return CheckContext(cfg, prims, clients)


def run_cases(ctx: CheckContext, specs, cases: Sequence[dict]) -> Dict[str, CheckResult]:
    return {s.key: run_check(s, ctx, save=False) for s in specs}


def expected_vs_actual(results: Dict[str, CheckResult], ctx: CheckContext, cases: Sequence[dict]) -> pd.DataFrame:
    """One row per test caption: the rows expected to flag, the rows that did, and whether they match."""
    flagged: Dict[str, set] = {sid: set() for sid in ctx.prims.segments.segment_id}
    reasons: Dict[str, List[str]] = {sid: [] for sid in flagged}
    for r in results.values():
        for f in r.findings.itertuples():
            if f.status in ("candidate", "report") and f.segment_id in flagged:
                flagged[f.segment_id].add(str(f.row))
                reasons[f.segment_id].append(f"row {f.row}: {f.message}")
    rows = []
    for sid, c in zip(ctx.prims.segments.segment_id, cases):
        exp = {x.strip() for x in str(c.get("expect", "")).split(",") if x.strip()}
        got = flagged[sid]
        rows.append({"caption": c["caption"], "expected rows": ", ".join(sorted(exp)) or "none",
                     "flagged rows": ", ".join(sorted(got)) or "none", "match": exp == got,
                     "why expected": c.get("why", ""), "messages": " / ".join(reasons[sid])})
    return pd.DataFrame(rows)
