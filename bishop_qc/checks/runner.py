"""Check registry and runner.

A check is a function that runs on one unit (a segment, a boundary or a recording) and returns
Findings. The runner calls it once per unit inside a cost ledger scope, so every unit gets its
own wall time and the API cost of the calls made inside it. API calls can run in parallel
threads; each thread keeps its own scope.

Each check declares the primitives it reads (uses). The cost report gives two numbers per
segment: the check alone (marginal), and the check plus those primitives (loaded).
"""
from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd

from ..costs import LEDGER, Ledger, loaded_cost, per_segment
from ..schema import Finding

Unit = Dict[str, Any]


@dataclass
class CheckSpec:
    key: str                                   # e.g., "row02_03"
    rows: str                                  # sheet rows, e.g., "2, 3"
    title: str
    category: str                              # e.g., "Segmentation errors / Boundary placement"
    uses: List[str]                            # primitives the check reads
    units: Callable[["CheckContext"], Iterable[Unit]]
    run: Callable[["CheckContext", Unit], List[Finding]]
    workers: int = 1                           # >1 for checks that wait on API calls
    unit: str = "segment"                      # what one unit is: segment, boundary or recording


@dataclass
class CheckContext:
    cfg: Any
    prims: Any
    clients: Any
    results: Dict[str, pd.DataFrame] = field(default_factory=dict)   # findings of earlier checks


@dataclass
class CheckResult:
    spec: CheckSpec
    findings: pd.DataFrame
    cost_per_unit: pd.DataFrame
    loaded: pd.DataFrame

    def status_counts(self) -> pd.DataFrame:
        return self.findings.groupby(["row", "status"]).size().unstack(fill_value=0)

    def cost_summary(self) -> pd.DataFrame:
        c = self.loaded
        if c.empty:
            return c
        return pd.DataFrame([{
            "check": self.spec.key, "rows": self.spec.rows, "units": len(c),
            "median_check_seconds": c.check_seconds.median(), "total_check_seconds": c.check_seconds.sum(),
            "total_check_seconds_this_run": c.check_seconds_this_run.sum(),
            "median_check_usd": c.check_usd.median(), "total_check_usd": c.check_usd.sum(),
            "median_loaded_seconds": c.loaded_seconds.median(), "median_loaded_usd": c.loaded_usd.median(),
            "api_calls": int(self.cost_per_unit.api_calls.sum()) if len(self.cost_per_unit) else 0,
        }])


REGISTRY: Dict[str, CheckSpec] = {}


def register(spec: CheckSpec) -> CheckSpec:
    REGISTRY[spec.key] = spec
    return spec


def findings_frame(findings: List[Finding]) -> pd.DataFrame:
    rows = []
    for f in findings:
        d = asdict(f)
        d["evidence"] = json.dumps(d["evidence"], default=str)
        rows.append(d)
    return pd.DataFrame(rows, columns=["row", "check", "recording_id", "segment_id", "status",
                                       "message", "evidence"])


def run_check(spec: CheckSpec, ctx: CheckContext, limit: Optional[int] = None,
              ledger: Ledger = LEDGER, save: bool = True) -> CheckResult:
    units = list(spec.units(ctx))
    if limit is not None:
        units = units[:limit]
    start = len(ledger.events)

    def one(u: Unit) -> List[Finding]:
        with ledger.scope(spec.key, u.get("recording_id", ""), u.get("segment_id", "")):
            return spec.run(ctx, u)

    if spec.workers > 1:
        with ThreadPoolExecutor(spec.workers) as pool:
            futures = [pool.submit(contextvars.copy_context().run, one, u) for u in units]
            out = [f.result() for f in futures]
    else:
        out = [one(u) for u in units]
    findings = findings_frame([f for fs in out for f in fs])
    events = ledger.frame().iloc[start:]
    cost = per_segment(events[events.name == spec.key], ctx.cfg)
    loaded = loaded_cost(cost, ctx.prims.cost_per_segment, {spec.key: spec.uses})
    ctx.results[spec.key] = findings
    if save:
        out_dir = Path(ctx.cfg.out_dir)
        (out_dir / "checks").mkdir(parents=True, exist_ok=True)
        (out_dir / "costs").mkdir(parents=True, exist_ok=True)
        findings.to_csv(out_dir / "checks" / f"{spec.key}_findings.csv", index=False)
        loaded.to_csv(out_dir / "costs" / f"{spec.key}_cost_per_unit.csv", index=False)
    return CheckResult(spec, findings, cost, loaded)


def evidence(findings: pd.DataFrame) -> pd.DataFrame:
    """Expand the evidence JSON into columns."""
    ev = pd.json_normalize([json.loads(e) for e in findings.evidence])
    return pd.concat([findings.drop(columns="evidence").reset_index(drop=True), ev], axis=1)
