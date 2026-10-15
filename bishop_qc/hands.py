"""P3 hand timeline: WiLoR hand boxes for side, SAM 3 hand masks for contact.

Hand boxes and side come from the hand detector shipped with WiLoR (an Ultralytics YOLOv8
pose model with the classes left and right), downloaded from the Hugging Face Hub. Its
license is CC BY-NC 4.0, so it suits this demo but not commercial delivery as is.
The detector has no ego or other person label. owner is a routing signal only: "ego" when the
box reaches the bottom edge of the frame, where a head camera sees the wearer's arms, and
"unclear" otherwise.

Contact comes from P6: SAM 3 segments "hand" and the caption nouns, and contact is the share
of the hand mask, grown by a few pixels, that touches an object mask. When no SAM 3 hand mask
matches a box, the box itself stands in for the mask. A hand that is not detected is unknown.
It is never "no contact".

Alternative: EgoHOS (Zhang et al., ECCV 2022, MIT license) segments the left hand, the right
hand and the object each hand touches, trained on egocentric video. It needs torch 1.11 and
mmcv-full 1.6 in a separate Python 3.10 environment. See the notebook, section 5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_hand_detector(cfg) -> Path:
    """Download the WiLoR hand detector once and cache it."""
    weights = Path(cfg.hand_detector_weights)
    if weights.exists():
        return weights
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=cfg.hand_detector_repo, filename=cfg.hand_detector_file,
                           local_dir=str(weights.parent))
    path = Path(path)
    if path != weights:
        weights.parent.mkdir(parents=True, exist_ok=True)
        path.replace(weights)
    return weights


def detect_hands(frames: pd.DataFrame, weights: Path, device: str, conf: float,
                 bottom_edge: float = 0.97, batch: int = 32, log=None) -> pd.DataFrame:
    """Every hand box on every sampled frame. Coordinates are fractions of the frame."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = {int(k): str(v).lower() for k, v in model.names.items()}
    unknown = set(names.values()) - {"left", "right"}
    if unknown:
        raise ValueError(f"Hand detector classes are {names}. Expected left and right.")
    rows = []
    recs = frames.to_dict("records")
    for i in range(0, len(recs), batch):
        if log and i and (i // batch) % 20 == 0:
            log(f"    {i}/{len(recs)} frames")
        chunk = recs[i:i + batch]
        results = model.predict([r["path"] for r in chunk], device=device, conf=conf, verbose=False)
        for r, res in zip(chunk, results):
            for box in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxyn[0].tolist()]
                rows.append({"recording_id": r["recording_id"], "frame_idx": r["frame_idx"], "t": r["t"],
                             "side": names[int(box.cls)],
                             "owner": "ego" if y2 >= bottom_edge else "unclear",
                             "conf": float(box.conf), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    cols = ["recording_id", "frame_idx", "t", "side", "owner", "conf", "x1", "y1", "x2", "y2"]
    return pd.DataFrame(rows, columns=cols)


def ego_hands(dets: pd.DataFrame) -> pd.DataFrame:
    """Best detection per frame and side, preferring boxes that reach the bottom edge."""
    d = dets.assign(_ego=(dets.owner == "ego").astype(int))
    d = d.sort_values(["_ego", "conf"], ascending=False)
    return d.drop_duplicates(["recording_id", "frame_idx", "side"]).drop(columns="_ego").reset_index(drop=True)


def _min_run(flags: np.ndarray, n: int) -> np.ndarray:
    """Keep True only in runs of at least n."""
    out = np.zeros_like(flags)
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            if j - i >= n:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def hand_timeline(frames: pd.DataFrame, dets: pd.DataFrame, overlap: pd.DataFrame, cfg) -> pd.DataFrame:
    """One row per timeline frame and side (left, right).

    state: unknown (not detected), contact, no_contact, or detected (no SAM 3 frame near it).
    """
    tl = frames[frames.is_timeline][["recording_id", "frame_idx", "t", "segment_id"]]
    ego = ego_hands(dets) if len(dets) else pd.DataFrame(
        columns=["recording_id", "frame_idx", "side", "owner", "conf", "x1", "y1", "x2", "y2"])
    rows = []
    for side in ("left", "right"):
        s = tl.merge(ego[ego.side == side][["recording_id", "frame_idx", "owner", "conf",
                                            "x1", "y1", "x2", "y2"]],
                     on=["recording_id", "frame_idx"], how="left")
        s["side"] = side
        rows.append(s)
    hl = pd.concat(rows, ignore_index=True)
    hl["detected"] = hl.conf.notna()

    if len(overlap):
        best = (overlap.sort_values("overlap", ascending=False)
                .drop_duplicates(["recording_id", "frame_idx", "side"])
                [["recording_id", "frame_idx", "side", "prompt", "obj_id", "overlap", "mask_source"]])
        best = best.rename(columns={"frame_idx": "sam_frame_idx", "prompt": "contact_object",
                                    "obj_id": "contact_obj_id"})
        out = []
        for (rid, side), grp in hl.groupby(["recording_id", "side"]):
            b = best[(best.recording_id == rid) & (best.side == side)]
            grp = grp.sort_values("t")
            if len(b):
                # The nearest SAM 3 frame within 0.25 s carries its overlap to the timeline frame.
                times = frames[frames.recording_id == rid].set_index("frame_idx")["t"]
                b = b.assign(sam_t=times.loc[b.sam_frame_idx].to_numpy()).sort_values("sam_t")
                grp = pd.merge_asof(grp, b.drop(columns=["recording_id", "side"]), left_on="t",
                                    right_on="sam_t", direction="nearest", tolerance=0.25)
            out.append(grp)
        hl = pd.concat(out, ignore_index=True)
    for col in ("contact_object", "contact_obj_id", "overlap", "sam_frame_idx", "mask_source"):
        if col not in hl:
            hl[col] = np.nan

    hl = hl.sort_values(["recording_id", "side", "t"]).reset_index(drop=True)
    contact = np.zeros(len(hl), dtype=bool)
    for _, idx in hl.groupby(["recording_id", "side"]).groups.items():
        idx = np.asarray(idx)
        raw = (hl.loc[idx, "overlap"].fillna(0) >= cfg.contact_overlap) & hl.loc[idx, "detected"]
        contact[idx] = _min_run(raw.to_numpy(), cfg.contact_min_frames)
    hl["state"] = np.select(
        [~hl.detected, contact, hl.overlap.notna()],
        ["unknown", "contact", "no_contact"], default="detected")
    hl.loc[hl.state != "contact", ["contact_object", "contact_obj_id"]] = np.nan
    return hl
