"""Section 9: one table over every check, what a human does with its output, and what it costs."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .report import check_counts
from .runner import CheckResult

ROLES = {
    "rule": "Rule: a flag is an error, the annotator fixes it",
    "model": "Model or signal: a flag is a candidate, a reviewer confirms",
    "report": "Report: measured for the client or a reviewer, not scored",
    "gap": "Not covered by the inputs: needs a gold set",
}
RULE_KEYS = {"row05", "row07", "row19", "row35", "row38", "row51", "row56", "row59", "row66", "row67", "row72",
             "row73", "row75", "row78", "row79", "row80", "row81", "row82"}
REPORT_KEYS = {"row16", "row43", "row53", "row69", "row71", "row74", "row77", "row83"}
GAP_KEYS = {"row10"}
MODEL_NAMES = {"gpt-6.1-sol": "Sol", "gpt-6-luna": "Decisions", "jev": "Jev"}


def role(key: str) -> str:
    return "gap" if key in GAP_KEYS else "report" if key in REPORT_KEYS else "rule" if key in RULE_KEYS else "model"


def _models(events: pd.DataFrame, key: str) -> str:
    e = events[(events.name == key) & (events.kind == "api")]
    names = sorted({MODEL_NAMES.get(str(m), "Jev" if str(m).startswith("jev") else str(m)) for m in e.model.dropna()})
    return ", ".join(names) or "none"


def check_overview(results: Dict[str, CheckResult], examples: pd.DataFrame, events: pd.DataFrame,
                   video_hours: float) -> pd.DataFrame:
    rows = []
    for key, r in results.items():
        c = check_counts(r)
        assessed = c["pass"] + c["candidate"]
        ex = examples.loc[key] if key in examples.index else None
        rows.append({
            "check": key, "rows": r.spec.rows, "category": r.spec.category.split(" / ")[-1],
            "title": r.spec.title, "role": role(key), "primitives": ", ".join(r.spec.uses) or "none",
            "models": _models(events, key), "unit": r.spec.unit, "units": int(sum(c.values())),
            "passed": c["pass"], "flagged": c["candidate"], "could not assess": c["unassessable"],
            "reported": c["report"], "flag rate": round(c["candidate"] / assessed, 3) if assessed else None,
            "usd per video hour": r.loaded.check_usd.sum() / video_hours if len(r.loaded) else 0.0,
            "seconds per video hour": r.loaded.check_seconds.sum() / video_hours if len(r.loaded) else 0.0,
            "reviewed example": ex.verdict if ex is not None else "",
        })
    return pd.DataFrame(rows).set_index("check")


def category_overview(ov: pd.DataFrame) -> pd.DataFrame:
    g = ov.groupby("category", sort=False)
    out = pd.DataFrame({
        "checks": g.size(),
        "rule": g.role.apply(lambda s: int((s == "rule").sum())),
        "model": g.role.apply(lambda s: int((s == "model").sum())),
        "report": g.role.apply(lambda s: int((s == "report").sum())),
        "flagged units": g.flagged.sum(),
        "could not assess": g["could not assess"].sum(),
        "usd per video hour": g["usd per video hour"].sum(),
        "seconds per video hour": g["seconds per video hour"].sum(),
    })
    out.loc["All checks"] = out.sum(numeric_only=True)
    return out


def headline(ov: pd.DataFrame, n_segments: int, flagged_segments: int, video_hours: float, primitives_usd_per_hour: float,
             gpu_ran: bool) -> str:
    """The opening lines of section 9, in plain sentences."""
    r = ov.role.value_counts()
    cands = int(ov.flagged.sum())
    checks_usd = float(ov["usd per video hour"].sum())
    lines = [
        f"This run checked {n_segments} segments, {video_hours * 60:.1f} minutes of video, with {len(ov)} checks.",
        f"{r.get('rule', 0)} checks are rules. {r.get('model', 0)} checks raise candidates for a reviewer. "
        f"{r.get('report', 0)} checks report a number without a score. {r.get('gap', 0)} check looks for an error "
        "that the inputs cannot show.",
        f"The checks flagged {cands} units, about {cands / video_hours:.0f} per hour of video. {flagged_segments} of "
        f"{n_segments} segments have at least one flag.",
    ]
    if gpu_ran:
        lines.append(f"The GPU primitives ran. Per hour of video, the primitives cost ${primitives_usd_per_hour:.2f} and "
                     f"the checks cost ${checks_usd:.2f}. The total is ${primitives_usd_per_hour + checks_usd:.2f}.")
    else:
        lines.append(f"The GPU primitives did not run, because this run had no GPU. Per hour of video, the primitives "
                     f"that ran cost ${primitives_usd_per_hour:.2f} and the checks cost ${checks_usd:.2f}. The GPU time "
                     "is missing from these totals.")
    return "\n\n".join(lines)
