"""Run settings, device selection and API keys."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

API_KEY_NAMES = ("OPENAI_API_KEY", "JEV_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN")
ENV_FILES = (".env.local", ".env")


@dataclass
class Config:
    data_dir: Path
    out_dir: Path
    cache_dir: Path

    # Guideline values (Bishop v2.0)
    boundary_tolerance_s: float = 0.1
    min_segment_s: float = 0.2
    idle_pause_s: float = 1.0
    # The CSV stores times to 0.1 s, so rounding adds up to half a step.
    csv_time_resolution_s: float = 0.1
    # Export gate tolerance. Not the annotation tolerance.
    gate_epsilon_s: float = 0.001

    # Frame sampling
    timeline_fps: float = 10.0
    boundary_window_s: float = 0.5
    frame_width: int = 640
    jpeg_quality: int = 90

    # Caption parser
    spacy_model: str = "en_core_web_sm"

    # Hands: WiLoR hand detector (YOLOv8, classes left and right). CC BY-NC 4.0.
    hand_detector_repo: str = "warmshao/WiLoR-mini"
    hand_detector_file: str = "pretrained_models/detector.pt"
    hand_detector_weights: Optional[Path] = None
    hand_conf: float = 0.3
    # Contact is a heuristic: share of the SAM 3 hand mask, grown by contact_dilate_px,
    # that touches an object mask. Calibrate on the gold set.
    contact_dilate_px: int = 6
    contact_overlap: float = 0.05
    contact_min_frames: int = 2

    # Motion (RAFT)
    raft_size: Tuple[int, int] = (384, 216)
    onset_mad_k: float = 3.0

    # Objects (SAM 3)
    sam3_checkpoint: str = "facebook/sam3"
    # SAM 3 is the slowest primitive. On a T4, sam3_fps = 2 and sam3_image_size = 560 cut its time
    # several times over (fewer frames, about a third of the pixels), at some cost in accuracy:
    # the model is trained at 1008 px.
    sam3_fps: float = 5.0
    sam3_image_size: Optional[int] = None   # None keeps the native 1008 px
    sam3_max_prompts: int = 6
    sam3_extra_prompts: Tuple[str, ...] = ("hand", "person")

    # API models
    sol_model: str = "gpt-6.1-sol"
    sol_reasoning_effort: str = "low"
    # OpenAI Decisions API: request format confirmed against the live API (see clients.py).
    decisions_model: str = os.getenv("OPENAI_DECISIONS_MODEL", "gpt-6-luna")
    decisions_url: str = os.getenv("OPENAI_DECISIONS_URL", "https://api.openai.com/v1/decisions")
    # Jev: TypeSafe's own API when JEV_API_KEY is set, else OpenRouter with OPENROUTER_API_KEY.
    jev_model: str = os.getenv("JEV_MODEL", "jev-latest")
    jev_url: str = "https://api.typesafe.ai/v1/systemone"
    jev_openrouter_model: str = "typesafe/jev-1.14"
    jev_openrouter_url: str = "https://openrouter.ai/api/alpha/decisions"

    # Checks
    # Clip questions (several frames): "sol" or "decisions" (gpt-6-luna), per check.
    # Chosen from the model comparison in the notebook, section 8.2.
    clip_model: Dict[str, str] = field(default_factory=lambda: {
        "row08": "sol", "row11": "decisions", "row12": "decisions", "row14": "decisions"})
    image_flag_prob: float = 0.7        # an image answer at or above this raises a flag
    boundary_audit_rate: float = 0.05   # share of unflagged boundaries sent to review anyway

    # Cost rates. ASSUMPTIONS: replace with the real hourly rate of the machine you run on.
    gpu_usd_per_hour: float = 0.50
    cpu_usd_per_hour: float = 0.05

    # Sample videos are not in git (GitHub rejects files over 100 MB). They are fetched from this
    # public Google Drive folder when missing, then checked against SampleInput/videos.sha256.
    videos_drive_folder: str = os.getenv(
        "BISHOP_VIDEOS_DRIVE", "https://drive.google.com/drive/folders/1sy4dCwklWqDzdzylXIhqztAOk_lkOGsC")

    # Synthetic platform fields for the sample data
    synthesize_platform_fields: bool = True
    seed: int = 7

    device: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def boundary_tolerance_effective_s(self) -> float:
        return self.boundary_tolerance_s + self.csv_time_resolution_s / 2

    def ensure_dirs(self) -> None:
        for p in (self.out_dir, self.cache_dir, self.out_dir / "timelines"):
            Path(p).mkdir(parents=True, exist_ok=True)


def make_config(root: Path, data_dir: Optional[Path] = None, **overrides) -> Config:
    root = Path(root)
    cfg = Config(
        data_dir=Path(data_dir) if data_dir else root / "SampleInput",
        out_dir=root / "outputs",
        cache_dir=root / "cache",
        **overrides,
    )
    if cfg.hand_detector_weights is None:
        env = os.getenv("HAND_DETECTOR_WEIGHTS")
        cfg.hand_detector_weights = Path(env) if env else cfg.cache_dir / "models" / "wilor_hand_detector.pt"
    cfg.ensure_dirs()
    return cfg


def select_device() -> str:
    """CUDA first, then Apple MPS. Fail with a clear note when neither exists."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    raise RuntimeError(
        "No GPU found. The primitives need CUDA or Apple MPS. "
        "In Colab, open Runtime > Change runtime type and pick a GPU, then rerun this cell."
    )


def read_env_file(path: Path) -> Dict[str, str]:
    """KEY=VALUE lines. Ignores comments and blank lines, strips quotes and "export"."""
    out: Dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.replace("export ", "").strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            out[key] = value
    return out


def load_keys(names: Iterable[str] = API_KEY_NAMES, root: Optional[Path] = None) -> Dict[str, bool]:
    """Read keys from the environment, then .env.local or .env in root, then Colab secrets.

    A key already in the environment is never overwritten. Values are never printed.
    Returns which keys are set.
    """
    if root is not None:
        for fname in ENV_FILES:
            p = Path(root) / fname
            if p.exists():
                for k, v in read_env_file(p).items():
                    os.environ.setdefault(k, v)
    status = {}
    for name in names:
        value = os.getenv(name)
        if not value:
            try:
                from google.colab import userdata  # type: ignore

                value = userdata.get(name)
            except Exception:
                value = None
        if value:
            os.environ[name] = value
        status[name] = bool(value)
    return status
