"""Time and cost ledger for primitives and checks.

Every event carries a name (a primitive such as "P6" or a check such as "row31"), the
recording and segment it ran on, the wall time in seconds, and the API cost in USD.

API cost comes from the token usage each response reports, priced with API_PRICING.
A disk cache hit costs nothing this run and takes no time, so each API event also keeps
usd_uncached and seconds_uncached: the price and the latency of the same call when it is not
cached. Those are the numbers to use for budgeting.

Compute cost is wall time times an hourly rate. The rates in Config are ASSUMPTIONS:
replace them with the real rate of the machine that runs the pipeline.

Work done once per recording (decoding frames, running the hand detector over all frames) is
attributed to segments in proportion to their frame count.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Mapping, Optional

import pandas as pd

_SCOPE: ContextVar[Dict[str, Any]] = ContextVar("bishop_cost_scope", default={})

# USD per 1M tokens.
# gpt-6.1-sol: OpenAI's GPT-6.1 Sol announcement (standard tier).
# gpt-6-luna: third-party price trackers (eesel.ai, anotherwrapper.com, October 2026). Not
#   confirmed on OpenAI's pricing page.
# jev: third-party write-ups (eesel.ai, flaviocopes.com, layer3labs.io): input only, output
#   free. TypeSafe's own pricing page was not reachable. Used only when usage has no cost.
API_PRICING = {
    "gpt-6.1-sol": {"input": 2.00, "cached_input": 0.10, "output": 10.00},
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    "jev": {"input": 0.042, "cached_input": 0.042, "output": 0.0},
}

EVENT_COLUMNS = ["name", "kind", "recording_id", "segment_id", "seconds", "seconds_uncached", "usd",
                 "usd_uncached", "device", "model", "tokens_in", "tokens_out", "cached", "reused"]


def api_usd(model: str, usage: Optional[Mapping[str, Any]]) -> float:
    """Price one API call from its usage block. NaN when the model has no price."""
    price = API_PRICING.get(model)
    if not price or not usage:
        return float("nan")
    tin = float(usage.get("input_tokens", 0) or 0)
    cached = float((usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0)
    tout = float(usage.get("output_tokens", 0) or 0)
    return ((tin - cached) * price["input"] + cached * price["cached_input"] + tout * price["output"]) / 1e6


class Ledger:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def _add(self, **event) -> None:
        row = {c: None for c in EVENT_COLUMNS}
        row.update({k: v for k, v in _SCOPE.get().items() if k in EVENT_COLUMNS})
        row.update(event)
        self.events.append(row)

    @contextmanager
    def scope(self, name: str, recording_id: str = "", segment_id: str = "",
              device: str = "cpu") -> Iterator[None]:
        """Time a block of work. API calls inside it are attributed to the same name and segment."""
        token = _SCOPE.set({"name": name, "recording_id": recording_id, "segment_id": segment_id,
                            "device": device})
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._add(kind="compute", seconds=dt, seconds_uncached=dt, usd=0.0, usd_uncached=0.0,
                      cached=False, reused=False)
            _SCOPE.reset(token)

    def api(self, model: str, seconds: float, usd: float, usd_uncached: float,
            tokens_in: float = float("nan"), tokens_out: float = float("nan"), cached: bool = False,
            seconds_uncached: Optional[float] = None) -> None:
        self._add(kind="api", model=model, seconds=seconds, usd=usd, usd_uncached=usd_uncached,
                  seconds_uncached=seconds if seconds_uncached is None else seconds_uncached,
                  tokens_in=tokens_in, tokens_out=tokens_out, cached=cached, reused=False)

    def apportion(self, name: str, recording_id: str, seconds: float, weights: Mapping[str, float],
                  device: str = "cpu") -> None:
        """Split one recording-level timing across its segments by weight (e.g., frame count)."""
        total = float(sum(weights.values())) or 1.0
        for sid, w in weights.items():
            self.events.append({**{c: None for c in EVENT_COLUMNS}, "name": name, "kind": "compute",
                                "recording_id": recording_id, "segment_id": sid,
                                "seconds": seconds * w / total, "seconds_uncached": seconds * w / total,
                                "usd": 0.0, "usd_uncached": 0.0,
                                "device": device, "cached": False, "reused": False})

    def extend(self, rows: pd.DataFrame) -> None:
        """Reuse timings saved by an earlier run, for stages loaded from cache."""
        for r in rows.to_dict("records"):
            r["reused"] = True
            self.events.append({c: r.get(c) for c in EVENT_COLUMNS})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events, columns=EVENT_COLUMNS)

    def reset(self) -> None:
        self.events.clear()


LEDGER = Ledger()


def per_segment(events: pd.DataFrame, cfg) -> pd.DataFrame:
    """Seconds and USD per name and segment. Compute USD uses the hourly rates in cfg.

    seconds is the wall time of this run. seconds_uncached adds back the latency of API calls
    that came from the cache, so it is the time the work takes on a fresh run.
    """
    if events.empty:
        return pd.DataFrame(columns=["name", "segment_id", "seconds", "seconds_uncached", "api_usd",
                                     "api_usd_uncached", "compute_usd", "total_usd", "api_calls"])
    e = events.copy()
    if "seconds_uncached" not in e:
        e["seconds_uncached"] = e["seconds"]
    su = pd.to_numeric(e.seconds_uncached, errors="coerce")
    s = pd.to_numeric(e.seconds, errors="coerce")
    e["_extra"] = ((e.kind == "api") * (su.fillna(s) - s)).fillna(0)
    rate = e.device.map(lambda d: cfg.gpu_usd_per_hour if d in ("cuda", "mps") else cfg.cpu_usd_per_hour)
    e["compute_usd"] = (e.kind == "compute") * e.seconds.astype(float) * rate.astype(float) / 3600
    g = e.groupby(["name", "segment_id"], dropna=False).agg(
        seconds=("seconds", "sum"),
        _extra=("_extra", "sum"),
        api_usd=("usd", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        api_usd_uncached=("usd_uncached", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        compute_usd=("compute_usd", "sum"),
        api_calls=("kind", lambda s: int((s == "api").sum())),
    ).reset_index()
    g["seconds_uncached"] = g.seconds + g.pop("_extra")
    g["total_usd"] = g.api_usd_uncached + g.compute_usd
    return g


def loaded_cost(check_costs: pd.DataFrame, primitive_costs: pd.DataFrame,
                uses: Mapping[str, List[str]]) -> pd.DataFrame:
    """Check cost per segment plus the share of each primitive it reads.

    uses maps a check name to the primitives it depends on, e.g., {"row31": ["P5", "P2", "P6"]}.
    A primitive shared by several checks is counted in full for each one. That gives the cost to
    run the check alone, not a split of a shared bill.
    """
    rows = []
    prim = primitive_costs.set_index(["name", "segment_id"])
    for r in check_costs.itertuples():
        sec, usd = float(r.seconds_uncached), float(r.total_usd)
        for p in uses.get(r.name, []):
            if (p, r.segment_id) in prim.index:
                sec += float(prim.loc[(p, r.segment_id), "seconds_uncached"])
                usd += float(prim.loc[(p, r.segment_id), "total_usd"])
        rows.append({"name": r.name, "segment_id": r.segment_id, "check_seconds": r.seconds_uncached,
                     "check_seconds_this_run": r.seconds, "check_usd": r.total_usd,
                     "loaded_seconds": sec, "loaded_usd": usd})
    return pd.DataFrame(rows)
