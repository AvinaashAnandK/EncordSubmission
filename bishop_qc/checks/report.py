"""Plain language results: per check, and per segment across a set of checks."""
from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd

from .runner import CheckResult

PLURAL = {"segment": "segments", "boundary": "boundaries", "recording": "recordings"}
STATUS_WORDS = {
    "pass": "passed",
    "candidate": "flagged for review",
    "unassessable": "could not be assessed",
    "report": "reported as a measurement (no verdict)",
    "no_coverage": "not covered by the inputs",
}


def check_counts(result: CheckResult) -> Dict[str, int]:
    vc = result.findings.status.value_counts()
    return {s: int(vc.get(s, 0)) for s in STATUS_WORDS}


def check_sentence(result: CheckResult) -> str:
    """One sentence: how many units passed, and how many need attention."""
    spec, c = result.spec, check_counts(result)
    total = int(sum(c.values()))
    units = PLURAL.get(spec.unit, spec.unit + "s")
    if total == 0:
        return f"Row {spec.rows} ({spec.title}): no {units} of this kind in the data, so nothing to check."
    parts = [f"Row {spec.rows} ({spec.title}): {c['pass']} of {total} {units} passed."]
    for s in ("candidate", "unassessable", "report", "no_coverage"):
        if c[s]:
            parts.append(f"{c[s]} {STATUS_WORDS[s]}.")
    return " ".join(parts)


def flagged_table(result: CheckResult, statuses=("candidate",)) -> pd.DataFrame:
    """One row per flagged unit, with the segment(s) it touches and the reason."""
    f = result.findings[result.findings.status.isin(statuses)]
    rows = []
    for r in f.itertuples():
        ev = json.loads(r.evidence) if isinstance(r.evidence, str) else {}
        unit = r.segment_id or r.recording_id
        if result.spec.unit == "boundary" and ev.get("prev_id"):
            unit = f"{ev['prev_id']} | {r.segment_id}"
        rows.append({"row": r.row, result.spec.unit: unit, "status": STATUS_WORDS[r.status], "reason": r.message})
    return pd.DataFrame(rows)


def unassessable_reasons(result: CheckResult) -> pd.DataFrame:
    f = result.findings[result.findings.status.isin(["unassessable", "no_coverage"])]
    if f.empty:
        return pd.DataFrame(columns=["reason", "units"])
    return f.groupby("message").size().rename("units").reset_index().rename(columns={"message": "reason"})


def summary_table(results: Dict[str, CheckResult]) -> pd.DataFrame:
    rows = []
    for key, r in results.items():
        c = check_counts(r)
        total = int(sum(c.values()))
        assessed = c["pass"] + c["candidate"]
        rows.append({
            "check": key, "rows": r.spec.rows, "title": r.spec.title, "unit": r.spec.unit, "units": total,
            "passed": c["pass"], "flagged": c["candidate"], "could not assess": c["unassessable"],
            "reported": c["report"], "not covered": c["no_coverage"],
            "pass rate of assessed": f"{c['pass'] / assessed:.0%}" if assessed else "n/a",
        })
    return pd.DataFrame(rows).set_index("check")


def segment_rollup(results: Dict[str, CheckResult], segments: pd.DataFrame) -> pd.DataFrame:
    """Per segment: the worst status across the checks, and every reason it was flagged.

    A boundary finding counts for both segments that meet at the boundary. Recording level
    findings (no segment) are left out.
    """
    hits: Dict[str, List[tuple]] = {sid: [] for sid in segments.segment_id}
    for key, r in results.items():
        for f in r.findings.itertuples():
            ev = json.loads(f.evidence) if isinstance(f.evidence, str) else {}
            sids = [s for s in (ev.get("prev_id") if r.spec.unit == "boundary" else None, f.segment_id) if s]
            for sid in sids:
                if sid in hits:
                    hits[sid].append((f.status, f"row {f.row}: {f.message}"))
    rank = {"candidate": 0, "unassessable": 1, "report": 2, "no_coverage": 3, "pass": 4}
    rows = []
    for sid, hs in hits.items():
        statuses = [h[0] for h in hs]
        worst = min(statuses, key=lambda s: rank[s]) if statuses else "pass"
        flagged = [m for s, m in hs if s == "candidate"]
        pending = [m for s, m in hs if s == "unassessable"]
        rows.append({"segment_id": sid, "result": {"candidate": "flagged", "unassessable": "passed so far, some checks pending",
                                                   "report": "passed", "no_coverage": "passed", "pass": "passed"}[worst],
                     "flags": len(flagged), "pending": len(pending), "reasons": " / ".join(flagged)})
    return pd.DataFrame(rows)


def rollup_sentence(rollup: pd.DataFrame, section: str) -> str:
    n = len(rollup)
    passed = int((rollup.result == "passed").sum())
    pending = int((rollup.result == "passed so far, some checks pending").sum())
    flagged = int((rollup.result == "flagged").sum())
    return (f"Section {section}: {passed} of {n} segments passed every check. {flagged} were flagged by at least one "
            f"check. {pending} passed every check that could run, with at least one check waiting for the GPU primitives.")
