"""P4 motion timeline: RAFT optical flow, split into camera motion and hand motion.

The camera motion comes from an affine fit to the flow outside the hand boxes.
pan_x and pan_y are the flow at the frame centre, in frame widths per second. A turn of the
head or body shows as pan_x. divergence is the expansion of the flow field, so walking
forward shows as a positive divergence. All values are relative. None of them is an angle.
hand_motion is the flow left after the camera fit, inside each ego hand box.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class RaftFlow:
    def __init__(self, device: str, size: Tuple[int, int]):
        import torch
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        self.torch = torch
        self.device = device
        self.size = size  # (width, height), both divisible by 8
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights, progress=False).to(device).eval()
        self.transforms = weights.transforms()

    def _load(self, paths: List[str]):
        import cv2

        imgs = []
        for p in paths:
            im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, self.size, interpolation=cv2.INTER_AREA)
            imgs.append(self.torch.from_numpy(im).permute(2, 0, 1))
        return self.torch.stack(imgs)

    def flow(self, paths_a: List[str], paths_b: List[str]) -> np.ndarray:
        """Flow from a to b, shape (N, 2, H, W), in pixels at self.size."""
        torch = self.torch
        a, b = self._load(paths_a), self._load(paths_b)
        a, b = self.transforms(a, b)
        with torch.inference_mode():
            out = self.model(a.to(self.device), b.to(self.device))[-1]
        return out.float().cpu().numpy()


def _fit_affine(flow: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Least squares affine model u = a0 + a1 x + a2 y, v = b0 + b1 x + b2 y on normalized x, y."""
    _, h, w = flow.shape
    ys, xs = np.mgrid[0:h:4, 0:w:4]
    keep = mask[ys, xs]
    x = (xs[keep] / (w - 1)) * 2 - 1
    y = (ys[keep] / (h - 1)) * 2 - 1
    A = np.stack([np.ones_like(x), x, y], axis=1)
    u, v = flow[0][ys, xs][keep], flow[1][ys, xs][keep]
    if len(A) < 10:
        return np.full(3, np.nan), np.full(3, np.nan)
    pu = np.linalg.lstsq(A, u, rcond=None)[0]
    pv = np.linalg.lstsq(A, v, rcond=None)[0]
    return pu, pv


def _box_mask(h: int, w: int, box) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = box
    m[int(y1 * h):int(np.ceil(y2 * h)), int(x1 * w):int(np.ceil(x2 * w))] = True
    return m


def _pairs(frames: pd.DataFrame) -> List[Tuple[str, dict, dict]]:
    """Consecutive frame pairs: the timeline track, then native frames near each boundary."""
    pairs = []
    tl = frames[frames.is_timeline].sort_values("t").to_dict("records")
    pairs += [("timeline", a, b) for a, b in zip(tl, tl[1:])]
    nb = frames[frames.near_boundary].sort_values("t").to_dict("records")
    for a, b in zip(nb, nb[1:]):
        if b["frame_idx"] == a["frame_idx"] + 1:
            pairs.append(("native", a, b))
    return pairs


def compute_motion(frames: pd.DataFrame, ego: pd.DataFrame, raft: RaftFlow, batch: int = 16) -> pd.DataFrame:
    """One row per frame pair of one recording."""
    boxes: Dict[Tuple[int, str], tuple] = {
        (int(r.frame_idx), r.side): (r.x1, r.y1, r.x2, r.y2) for r in ego.itertuples()}
    pairs = _pairs(frames)
    rows = []
    w, h = raft.size
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        flows = raft.flow([a["path"] for _, a, _ in chunk], [b["path"] for _, _, b in chunk])
        for (track, a, b), fl in zip(chunk, flows):
            dt = max(b["t"] - a["t"], 1e-3)
            fi = int(b["frame_idx"])
            hand_masks = {s: _box_mask(h, w, boxes[(fi, s)]) for s in ("left", "right") if (fi, s) in boxes}
            outside = ~np.any(np.stack(list(hand_masks.values())), axis=0) if hand_masks else np.ones((h, w), bool)
            pu, pv = _fit_affine(fl, outside)
            ys, xs = np.mgrid[0:h, 0:w]
            xn, yn = (xs / (w - 1)) * 2 - 1, (ys / (h - 1)) * 2 - 1
            resid = np.hypot(fl[0] - (pu[0] + pu[1] * xn + pu[2] * yn),
                             fl[1] - (pv[0] + pv[1] * xn + pv[2] * yn))
            row = {
                "recording_id": a["recording_id"], "track": track,
                "frame_idx_a": int(a["frame_idx"]), "frame_idx": fi, "t": b["t"], "dt": dt,
                "pan_x": pu[0] / w / dt, "pan_y": pv[0] / h / dt,
                # Affine terms are per normalized unit (half the frame), so scale to frame size.
                "divergence": (pu[1] / w + pv[2] / h) / dt,
                "curl": (pv[1] / w - pu[2] / h) / dt,
                "residual_bg": float(resid[outside].mean()) / w / dt,
            }
            for s in ("left", "right"):
                row[f"hand_motion_{s}"] = float(resid[hand_masks[s]].mean()) / w / dt if s in hand_masks else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _robust_threshold(x: pd.Series, k: float) -> float:
    x = x.dropna()
    if not len(x):
        return np.nan
    med = float(x.median())
    mad = float((x - med).abs().median()) * 1.4826
    return med + k * max(mad, 1e-6)


def motion_signals(motion: pd.DataFrame, k: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Add the three onset signals and return the per recording thresholds."""
    m = motion.copy()
    m["hand_motion"] = m[["hand_motion_left", "hand_motion_right"]].max(axis=1)
    m["camera_turn"] = m.pan_x.abs()
    m["forward"] = m.divergence
    tl = m[m.track == "timeline"]
    thr = {s: _robust_threshold(tl[s], k) for s in ("hand_motion", "camera_turn", "forward")}
    for s, v in thr.items():
        m[f"{s}_active"] = m[s] > v
    return m, thr


def onsets(motion: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Rising edges of each signal on the timeline track, after a 3 sample median filter."""
    rows = []
    tl = motion[motion.track == "timeline"].sort_values("t")
    for s, thr in thresholds.items():
        if not np.isfinite(thr):
            continue
        smooth = tl[s].rolling(3, center=True, min_periods=1).median()
        active = (smooth > thr).to_numpy()
        rise = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        rows += [{"recording_id": tl.recording_id.iloc[0], "t": float(tl.t.iloc[i]), "signal": s,
                  "value": float(smooth.iloc[i])} for i in rise]
    return pd.DataFrame(rows, columns=["recording_id", "t", "signal", "value"])
