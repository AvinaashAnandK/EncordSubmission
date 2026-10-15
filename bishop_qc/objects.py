"""P6 object tracks: SAM 3 prompted with the nouns of each caption.

Each segment is its own SAM 3 video session on its timeline frames, subsampled to sam3_fps.
The prompts are the nouns of the segment, the objects of its neighbors, and a few extra
prompts "hand" and "person". While the masks are in memory, this module matches each SAM 3
hand mask to a detector box (for its side) and measures how much of the hand mask, grown by a
few pixels, touches each object mask. P3 turns that overlap into contact.
A missed detection does not prove an object is out of view.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .costs import LEDGER
from .schema import ParsedCaption


def _short(phrase: str) -> str:
    """Drop an "of" tail and pronouns, e.g., "small bag of black parts" -> "small bag"."""
    phrase = phrase.split(" of ")[0].strip()
    return "" if phrase in ("it", "them", "hands", "hand") else phrase


def caption_nouns(pc: ParsedCaption) -> List[str]:
    out: List[str] = []
    for c in pc.clauses:
        for p in (c.object, c.place, c.place_from, c.held_object, c.tool, *c.landmarks):
            p = _short(p or "")
            if p and p not in out:
                out.append(p)
    return out


def segment_prompts(nouns_by_seg: Dict[str, List[str]], seg_ids: Sequence[str], i: int,
                    max_prompts: int, extra: Sequence[str]) -> List[str]:
    """This segment first, then the neighbors, then the extra prompts."""
    out: List[str] = []
    for j in (i, i - 1, i + 1):
        if 0 <= j < len(seg_ids):
            for p in nouns_by_seg.get(seg_ids[j], []):
                if p not in out:
                    out.append(p)
    out = out[:max_prompts]
    return out + [e for e in extra if e not in out]


class Sam3:
    def __init__(self, checkpoint: str, device: str, image_size=None):
        import torch
        from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if image_size:
            config = Sam3VideoConfig.from_pretrained(checkpoint)
            config.image_size = int(image_size)
            self.model = Sam3VideoModel.from_pretrained(checkpoint, config=config)
            self.processor = Sam3VideoProcessor.from_pretrained(
                checkpoint, size={"height": int(image_size), "width": int(image_size)})
        else:
            self.model = Sam3VideoModel.from_pretrained(checkpoint)
            self.processor = Sam3VideoProcessor.from_pretrained(checkpoint)
        self.model = self.model.to(device, dtype=self.dtype).eval()

    def track(self, images, prompts: List[str]):
        """Yield (local frame index, postprocessed outputs) for one clip."""
        session = self.processor.init_video_session(
            video=images, inference_device=self.device, processing_device="cpu",
            video_storage_device="cpu", dtype=self.dtype)
        self.processor.add_text_prompt(session, prompts)
        with self.torch.inference_mode():
            for out in self.model.propagate_in_video_iterator(inference_session=session):
                yield out.frame_idx, self.processor.postprocess_outputs(session, out)


HAND_PROMPT = "hand"
NOT_OBJECTS = {"hand", "person"}


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _box_mask(h: int, w: int, box) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = box
    m[int(y1 * h):int(np.ceil(y2 * h)), int(x1 * w):int(np.ceil(x2 * w))] = True
    return m


def _hand_masks(boxes, hand_objs, masks, h, w, min_iou: float = 0.3):
    """For each detector box (side), the SAM 3 hand mask with the best box IoU, else the box."""
    out = []
    for side, box in boxes:
        best, best_iou = None, min_iou
        for n, hbox in hand_objs:
            iou = _box_iou(box, hbox)
            if iou >= best_iou:
                best, best_iou = n, iou
        if best is None:
            out.append((side, _box_mask(h, w, box), "box"))
        else:
            out.append((side, masks[best], "sam3"))
    return out


def run_objects(sam: Sam3, frames: pd.DataFrame, seg: pd.DataFrame, parsed: Dict[str, ParsedCaption],
                ego: pd.DataFrame, cfg, log=print) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """SAM 3 on every segment of one recording. Returns (object rows, hand overlap rows)."""
    seg = seg.sort_values("segment_index")
    ids = list(seg.segment_id)
    nouns = {sid: caption_nouns(parsed[sid]) for sid in ids}
    step = max(1, int(round(cfg.timeline_fps / cfg.sam3_fps)))
    tl = frames[frames.is_timeline].sort_values("t")
    boxes: Dict[int, List[Tuple[str, tuple]]] = {}
    for r in ego.itertuples():
        boxes.setdefault(int(r.frame_idx), []).append((r.side, (r.x1, r.y1, r.x2, r.y2)))
    extra = tuple(cfg.sam3_extra_prompts) + ((HAND_PROMPT,) if HAND_PROMPT not in cfg.sam3_extra_prompts else ())
    import time

    obj_rows, ov_rows = [], []
    t_start, n = time.perf_counter(), len(seg)
    for i, row in enumerate(seg.itertuples()):
        clip = tl[tl.segment_id == row.segment_id].iloc[::step]
        if clip.empty:
            continue
        prompts = segment_prompts(nouns, ids, i, cfg.sam3_max_prompts, extra)
        with LEDGER.scope("P6", row.recording_id, row.segment_id, sam.device):
            _track_segment(sam, clip, row, prompts, boxes, cfg, obj_rows, ov_rows)
        done = i + 1
        if done % 10 == 0 or done == n:
            per = (time.perf_counter() - t_start) / done
            log(f"    {row.recording_id}: {done}/{n} segments, {per:.1f} s per segment, "
                f"about {per * (n - done) / 60:.1f} min left")
    obj_cols = ["recording_id", "segment_id", "frame_idx", "t", "prompt", "obj_id", "score",
                "x1", "y1", "x2", "y2", "mask_area"]
    ov_cols = ["recording_id", "frame_idx", "side", "prompt", "obj_id", "overlap", "mask_source"]
    return pd.DataFrame(obj_rows, columns=obj_cols), pd.DataFrame(ov_rows, columns=ov_cols)


def _track_segment(sam, clip, row, prompts, boxes, cfg, obj_rows, ov_rows) -> None:
    """SAM 3 on one segment. Appends object rows and hand overlap rows."""
    import cv2
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in clip.path]
    clip_recs = clip.to_dict("records")
    for k, out in sam.track(images, prompts):
        fr = clip_recs[k]
        h, w = images[k].height, images[k].width
        obj_to_prompt = {int(o): p for p, objs in out["prompt_to_obj_ids"].items() for o in objs}
        masks = out["masks"].cpu().numpy().astype(bool) if len(out["object_ids"]) else np.zeros((0, h, w), bool)
        hand_objs, objs = [], []
        for n, oid in enumerate(out["object_ids"].tolist()):
            x1, y1, x2, y2 = out["boxes"][n].tolist()
            prompt = obj_to_prompt.get(int(oid), "")
            nbox = (x1 / w, y1 / h, x2 / w, y2 / h)
            obj_rows.append({
                "recording_id": fr["recording_id"], "segment_id": row.segment_id,
                "frame_idx": fr["frame_idx"], "t": fr["t"], "prompt": prompt,
                "obj_id": f"{row.segment_id}:{oid}", "score": float(out["scores"][n]),
                "x1": nbox[0], "y1": nbox[1], "x2": nbox[2], "y2": nbox[3],
                "mask_area": float(masks[n].mean()),
            })
            if prompt == HAND_PROMPT:
                hand_objs.append((n, nbox))
            elif prompt not in NOT_OBJECTS:
                objs.append((n, prompt, f"{row.segment_id}:{oid}"))
        grow = np.ones((cfg.contact_dilate_px * 2 + 1,) * 2, np.uint8)
        for side, hmask, source in _hand_masks(boxes.get(int(fr["frame_idx"]), []), hand_objs, masks, h, w):
            area = hmask.sum()
            if not area:
                continue
            grown = cv2.dilate(hmask.astype(np.uint8), grow).astype(bool)
            for n, prompt, gid in objs:
                touch = (grown & masks[n] & ~hmask).sum()
                ov_rows.append({"recording_id": fr["recording_id"], "frame_idx": fr["frame_idx"],
                                "side": side, "prompt": prompt, "obj_id": gid,
                                "overlap": float(min(1.0, touch / area)), "mask_source": source})
