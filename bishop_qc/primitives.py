"""Run every primitive once, cache the results, and add the per segment columns to the CSV.

Order and dependencies:
  P1 video metadata
  P5 caption fields            (rules + spaCy, Sol fallback)
  P8 suggestion features       (needs P5)
  P9 edit time features
  P2 frames and blur           (needs P1)
  P3a hand boxes               (needs P2)
  P6 object tracks             (needs P2, P3a, P5)
  P3b hand timeline, contact   (needs P3a, P6)
  P4 motion timeline, onsets   (needs P2, P3a)
  P7 segment summary           (joins everything that ran)

Outputs, all under cfg.out_dir:
  recordings_enriched.csv, segments_enriched.csv, caption_clauses.csv
  timelines/frames.csv, hand_detections.csv, hand_timeline.csv, hand_object_overlap.csv,
  objects.csv, motion.csv, onsets.csv
  costs/primitive_events.csv, costs/primitive_cost_per_segment.csv

Every stage records its time per segment in the cost ledger. Work done once per recording is
split across segments by frame count. A stage loaded from cache reuses the timings saved by
the run that computed it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from . import parser as P
from .costs import LEDGER, Ledger, per_segment
from .platform_features import edit_time_features, suggestion_features
from .schema import ParsedCaption
from .summary import segment_summary
from .video import assign_segments, extract_frames, plan_frames, probe_video

ALL_STAGES = ("P1", "P5", "P8", "P9", "P2", "P3", "P6", "P4", "P7")
HEAVY_STAGES = ("P3", "P6", "P4")
EMPTY_EGO = ["recording_id", "frame_idx", "side", "owner", "conf", "x1", "y1", "x2", "y2"]


@dataclass
class Primitives:
    recordings: pd.DataFrame
    segments: pd.DataFrame
    parsed: Dict[str, ParsedCaption]
    clauses: pd.DataFrame
    frames: pd.DataFrame = field(default_factory=pd.DataFrame)
    hand_dets: pd.DataFrame = field(default_factory=pd.DataFrame)
    overlap: pd.DataFrame = field(default_factory=pd.DataFrame)
    objects: pd.DataFrame = field(default_factory=pd.DataFrame)
    hand_timeline: pd.DataFrame = field(default_factory=pd.DataFrame)
    motion: pd.DataFrame = field(default_factory=pd.DataFrame)
    onsets: pd.DataFrame = field(default_factory=pd.DataFrame)
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cost_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_per_segment: pd.DataFrame = field(default_factory=pd.DataFrame)


def parse_all(seg: pd.DataFrame, sol=None, ledger: Ledger = LEDGER):
    parsed: Dict[str, ParsedCaption] = {}
    for r in seg.itertuples():
        with ledger.scope("P5", r.recording_id, r.segment_id):
            parsed[r.segment_id] = P.parse(r.caption, sol)
    clause_rows = []
    for sid, pc in parsed.items():
        for c in pc.clauses:
            d = c.to_dict()
            d["attributes"] = json.dumps(d["attributes"])
            d["landmarks"] = json.dumps(d["landmarks"])
            clause_rows.append({"segment_id": sid, **d})
    clauses = pd.DataFrame(clause_rows)
    j = lambda xs: json.dumps([x for x in xs if x])
    cols = pd.DataFrame([{
        "segment_id": sid,
        "p5_source": pc.source,
        "p5_is_fixed_literal": pc.is_fixed_literal,
        "p5_literal": pc.literal,
        "p5_n_clauses": len(pc.clauses),
        "p5_verbs": j(c.verb_key for c in pc.clauses),
        "p5_objects": j(c.object for c in pc.clauses),
        "p5_actors": j(c.actor for c in pc.clauses),
        "p5_places": j(c.place for c in pc.clauses),
        "p5_landmarks": j(l for c in pc.clauses for l in c.landmarks),
        "p5_is_navigation": any(c.is_navigation for c in pc.clauses),
        "p5_has_failure_word": any(c.has_failure_word for c in pc.clauses),
        "p5_word_count": pc.word_count,
        "p5_comma_count": pc.comma_count,
        "p5_inner_sentence_ends": pc.sentence_ends,
        "p5_connectors": j(pc.connectors),
        "p5_warnings": j(pc.warnings),
    } for sid, pc in parsed.items()])
    return parsed, clauses, cols


def _frame_weights(frames: pd.DataFrame, rid: str, seg_ids) -> Dict[str, float]:
    f = frames[frames.recording_id == rid]
    counts = f.groupby("segment_id").size()
    return {sid: float(counts.get(sid, 0)) or 1.0 for sid in seg_ids}


def run_primitives(cfg, recordings: pd.DataFrame, segments: pd.DataFrame, clients=None,
                   stages: Iterable[str] = ALL_STAGES, device: Optional[str] = None,
                   force: bool = False, log=print, ledger: Ledger = LEDGER) -> Primitives:
    stages = set(stages)
    out_dir = Path(cfg.out_dir)
    tdir, cdir = out_dir / "timelines", out_dir / "costs"
    tdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)
    events_path = cdir / "primitive_events.csv"
    prev_events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame(columns=["name"])
    start_n = len(ledger.events)
    rec, seg = recordings.copy(), segments.copy()
    seg_ids = {rid: list(g.segment_id) for rid, g in seg.groupby("recording_id")}
    pts: Dict[str, np.ndarray] = {}

    pdir = tdir / "parts"
    pdir.mkdir(parents=True, exist_ok=True)

    def per_recording(name: str, fname: str, build_one: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
        """Build one recording at a time and save each part, so an interrupted run keeps its progress."""
        parts = []
        for rid in rec.recording_id:
            part = pdir / f"{fname}_{rid}.csv"
            if part.exists() and not force:
                log(f"  {rid}: loaded from cache")
                prev = prev_events[(prev_events.name == name)] if len(prev_events) else prev_events
                if len(prev) and "recording_id" in prev:
                    ledger.extend(prev[prev.recording_id == rid])
                parts.append(pd.read_csv(part))
                continue
            df = build_one(rid)
            df.to_csv(part, index=False)
            parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def cached(name: str, path: Path, build: Callable[[], pd.DataFrame]) -> Tuple[pd.DataFrame, bool]:
        if path.exists() and not force:
            log(f"  loaded {path.name} from cache")
            ledger.extend(prev_events[prev_events.name == name])
            return pd.read_csv(path), True
        df = build()
        df.to_csv(path, index=False)
        return df, False

    # P1
    log("P1 video metadata ...")
    meta = []
    for r in rec.itertuples():
        t0 = time.perf_counter()
        m = probe_video(r.path, cfg.boundary_tolerance_s, cfg.boundary_tolerance_effective_s)
        s = seg[seg.recording_id == r.recording_id]
        ledger.apportion("P1", r.recording_id, time.perf_counter() - t0,
                         dict(zip(s.segment_id, s.end_s - s.start_s)))
        pts[r.recording_id] = m.pop("_pts")
        meta.append({"recording_id": r.recording_id, **{f"p1_{k}": v for k, v in m.items()}})
    rec = rec.merge(pd.DataFrame(meta), on="recording_id", how="left")

    # P5
    log("P5 caption fields ...")
    sol = clients.sol if clients is not None and getattr(clients.sol, "available", False) else None
    parsed, clauses, p5 = parse_all(seg, sol, ledger)
    seg = seg.merge(p5, on="segment_id", how="left")

    # P8 and P9
    if "P8" in stages:
        log("P8 suggestion features ...")
        parts = []
        for i in range(len(seg)):
            row = seg.iloc[[i]]
            with ledger.scope("P8", row.recording_id.iloc[0], row.segment_id.iloc[0]):
                parts.append(suggestion_features(row, P.parse_caption))
        seg = seg.merge(pd.concat(parts, ignore_index=True), on="segment_id", how="left")
    if "P9" in stages:
        log("P9 edit time features ...")
        parts = []
        for rid, g in seg.groupby("recording_id"):
            t0 = time.perf_counter()
            parts.append(edit_time_features(g))
            ledger.apportion("P9", rid, time.perf_counter() - t0, {s: 1.0 for s in g.segment_id})
        seg = seg.merge(pd.concat(parts, ignore_index=True), on="segment_id", how="left")

    out = Primitives(rec, seg, parsed, clauses)

    # P2
    if "P2" in stages:
        log("P2 frames and blur ...")

        def build_frames():
            parts = []
            for r in rec.itertuples():
                t0 = time.perf_counter()
                s = seg[seg.recording_id == r.recording_id]
                bounds = np.unique(np.r_[s.start_s.to_numpy()[1:], s.end_s.to_numpy()[:-1]])
                plan = plan_frames(pts[r.recording_id], bounds, cfg.timeline_fps, cfg.boundary_window_s)
                fr = extract_frames(r.path, plan, Path(cfg.cache_dir) / "frames" / r.recording_id,
                                    cfg.frame_width, cfg.jpeg_quality)
                fr.insert(0, "recording_id", r.recording_id)
                fr = assign_segments(fr, s)
                parts.append(fr)
                ledger.apportion("P2", r.recording_id, time.perf_counter() - t0,
                                 _frame_weights(fr, r.recording_id, s.segment_id))
                log(f"  {r.recording_id}: {len(fr)} frames ({int(fr.is_timeline.sum())} timeline)")
            return pd.concat(parts, ignore_index=True)

        out.frames, _ = cached("P2", tdir / "frames.csv", build_frames)
        out.frames["segment_id"] = out.frames["segment_id"].fillna("")

    if stages & set(HEAVY_STAGES) and device is None:
        raise RuntimeError("Stages P3, P4 and P6 need a GPU device. Pass device=select_device().")

    # P3a
    if "P3" in stages:
        from .hands import detect_hands, ensure_hand_detector

        log("P3 hand boxes (WiLoR hand detector) ...")
        weights = ensure_hand_detector(cfg)

        def dets_one(rid):
            t0 = time.perf_counter()
            fr = out.frames[out.frames.recording_id == rid]
            log(f"  {rid}: {len(fr)} frames")
            d = detect_hands(fr, weights, device, cfg.hand_conf, log=log)
            ledger.apportion("P3", rid, time.perf_counter() - t0, _frame_weights(fr, rid, seg_ids[rid]), device)
            return d

        out.hand_dets, _ = cached("P3", tdir / "hand_detections.csv",
                                  lambda: per_recording("P3", "hand_detections", dets_one))

    from .hands import ego_hands

    ego = ego_hands(out.hand_dets) if len(out.hand_dets) else pd.DataFrame(columns=EMPTY_EGO)

    # P6
    if "P6" in stages:
        from .objects import Sam3, run_objects

        log("P6 object tracks (SAM 3) ...")

        sam_box = {}

        def objects_one(rid):
            if "sam" not in sam_box:
                log("  loading SAM 3 (the first run downloads several GB)")
                sam_box["sam"] = Sam3(cfg.sam3_checkpoint, device, cfg.sam3_image_size)
            o, v = run_objects(sam_box["sam"], out.frames[out.frames.recording_id == rid],
                               seg[seg.recording_id == rid], parsed, ego[ego.recording_id == rid], cfg, log=log)
            v.to_csv(pdir / f"hand_object_overlap_{rid}.csv", index=False)
            log(f"  {rid}: {len(o)} object rows")
            return o

        def build_objects():
            objs = per_recording("P6", "objects", objects_one)
            ovs = [pd.read_csv(p) for p in sorted(pdir.glob("hand_object_overlap_*.csv"))]
            pd.concat(ovs, ignore_index=True).to_csv(tdir / "hand_object_overlap.csv", index=False)
            return objs

        out.objects, _ = cached("P6", tdir / "objects.csv", build_objects)
        ov_path = tdir / "hand_object_overlap.csv"
        out.overlap = pd.read_csv(ov_path) if ov_path.exists() else pd.DataFrame()

    # P3b
    if "P3" in stages:
        from .hands import hand_timeline

        log("P3 hand timeline and contact ...")
        t0 = time.perf_counter()
        out.hand_timeline = hand_timeline(out.frames, out.hand_dets, out.overlap, cfg)
        out.hand_timeline.to_csv(tdir / "hand_timeline.csv", index=False)
        total = time.perf_counter() - t0
        n_all = max(len(out.frames), 1)
        for rid in rec.recording_id:
            share = (out.frames.recording_id == rid).sum() / n_all
            ledger.apportion("P3", rid, total * share, _frame_weights(out.frames, rid, seg_ids[rid]))

    # P4
    if "P4" in stages:
        from .motion import RaftFlow, compute_motion, motion_signals, onsets

        log("P4 motion timeline (RAFT) ...")

        raft_box = {}

        def motion_one(rid):
            if "raft" not in raft_box:
                raft_box["raft"] = RaftFlow(device, cfg.raft_size)
            t0 = time.perf_counter()
            fr = out.frames[out.frames.recording_id == rid]
            m = compute_motion(fr, ego[ego.recording_id == rid], raft_box["raft"], log=log)
            ledger.apportion("P4", rid, time.perf_counter() - t0, _frame_weights(fr, rid, seg_ids[rid]), device)
            log(f"  {rid}: {len(m)} frame pairs")
            return m

        raw, _ = cached("P4", tdir / "motion.csv", lambda: per_recording("P4", "motion", motion_one))
        sig, ons = [], []
        for rid, grp in raw.groupby("recording_id"):
            m, thr = motion_signals(grp, cfg.onset_mad_k)
            out.thresholds[rid] = thr
            sig.append(m)
            ons.append(onsets(m, thr))
        out.motion = pd.concat(sig, ignore_index=True)
        out.onsets = pd.concat(ons, ignore_index=True)
        out.onsets.to_csv(tdir / "onsets.csv", index=False)

    # P7
    if "P7" in stages:
        log("P7 segment summary ...")
        t0 = time.perf_counter()
        s7 = segment_summary(seg, parsed, out.frames, out.hand_dets, out.hand_timeline,
                             out.motion, out.objects, cfg)
        total = time.perf_counter() - t0
        for rid, ids in seg_ids.items():
            ledger.apportion("P7", rid, total * len(ids) / len(seg), {s: 1.0 for s in ids})
        seg = seg.merge(s7, on="segment_id", how="left")

    events = ledger.frame().iloc[start_n:]
    events = events[events.name.isin(ALL_STAGES)]
    fresh = events[events.reused != True]  # noqa: E712
    keep = prev_events[~prev_events.name.isin(fresh.name.unique())] if len(prev_events) else prev_events
    pd.concat([keep, fresh], ignore_index=True).to_csv(events_path, index=False)
    out.cost_events = events.reset_index(drop=True)
    out.cost_per_segment = per_segment(out.cost_events, cfg)
    out.cost_per_segment.to_csv(cdir / "primitive_cost_per_segment.csv", index=False)

    out.segments, out.recordings = seg, rec
    rec.to_csv(out_dir / "recordings_enriched.csv", index=False)
    seg.to_csv(out_dir / "segments_enriched.csv", index=False)
    clauses.to_csv(out_dir / "caption_clauses.csv", index=False)
    log(f"Wrote {out_dir / 'segments_enriched.csv'} with {seg.shape[1]} columns.")
    return out
