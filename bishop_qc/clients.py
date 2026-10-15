"""API clients for Sol, OpenAI Decisions and Jev.

Rules for every client:
1. Every call is cached on disk, keyed by the request.
2. A missing key, an HTTP error or an answer that cannot be read gives an unassessable
   Answer. It never gives a pass.
3. Every call is logged in the cost ledger with its time, tokens and USD.

Sol: gpt-6.1-sol on the OpenAI Responses API. Text and image input, no video, so a clip is
sent as sampled frames. Used for questions on several frames and as the parser fallback.

OpenAI Decisions: POST /v1/decisions with a decision model (gpt-6-luna). The request format
below was confirmed against the live API on 2026-10-15 (the public docs have no reference):
  {"model": "gpt-6-luna",
   "input": "text" or [{"role": "user", "content": [{"type": "input_text", ...},
                                                    {"type": "input_image", "image_url": ...}]}],
   "questions": [{"name", "type": "predicate", "instructions"},
                 {"name", "type": "choice", "instructions", "choices": [{"value", "description"}]},
                 {"name", "type": "score", "instructions", "levels": [{"label", "description"}]}]}
  -> {"answers": [{"name", "type", "probability"} | {"choice", "probabilities": [{"value",
      "probability"}], "confidence"} | {"score", "probabilities": [{"value", "label",
      "probability"}], "confidence"}], "usage": {...}}
Used for closed questions on one image. Falls back to Sol on any failure.

Jev: TypeSafe's own API (POST https://api.typesafe.ai/v1/systemone) when JEV_API_KEY is set,
else OpenRouter. Question types noul, choice and score. Used for closed questions on text.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .costs import LEDGER, api_usd
from .schema import Answer

ImageLike = Union[str, Path, bytes, np.ndarray]


@dataclass
class Question:
    """A typed question.

    noul: criteria is {"true": ..., "false": ...} or None. The answer is the probability of true.
    choice: criteria is {option: description}. The answer is one option.
    score: criteria is a list of ordered levels. The answer is a level index.
    """

    name: str
    type: str
    instructions: str
    criteria: Any = None

    def payload(self) -> Dict[str, Any]:
        crit = self.criteria
        if self.type == "noul" and crit is None:
            crit = {"true": "Yes.", "false": "No."}
        return {"type": self.type, "instructions": self.instructions, "criteria": crit}

    def decisions_payload(self) -> Dict[str, Any]:
        """The same question in the OpenAI Decisions format."""
        if self.type == "noul":
            text = self.instructions
            if self.criteria:
                text += f" True means: {self.criteria['true']} False means: {self.criteria['false']}"
            return {"name": self.name, "type": "predicate", "instructions": text}
        if self.type == "choice":
            return {"name": self.name, "type": "choice", "instructions": self.instructions,
                    "choices": [{"value": k, "description": v} for k, v in self.criteria.items()]}
        return {"name": self.name, "type": "score", "instructions": self.instructions,
                "levels": [{"label": str(v), "description": str(v)} for v in self.criteria]}


def image_bytes(img: ImageLike) -> bytes:
    if isinstance(img, (str, Path)):
        return Path(img).read_bytes()
    if isinstance(img, bytes):
        return img
    import cv2

    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("could not encode image")
    return buf.tobytes()


def data_uri(img: ImageLike) -> str:
    b = image_bytes(img)
    mime = "image/png" if b[:4] == b"\x89PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"


class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:32]

    def get(self, key: str) -> Optional[Any]:
        p = self.root / f"{key}.json"
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except (OSError, ValueError):          # a partial file from an interrupted write is a miss
            return None

    def put(self, key: str, value: Any) -> None:
        """Write to a temporary file, then rename, so a reader in another thread never sees half a file."""
        tmp = self.root / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps(value))
        os.replace(tmp, self.root / f"{key}.json")


def _image_ids(images: Sequence[ImageLike]) -> List[str]:
    return [hashlib.sha256(image_bytes(im)).hexdigest()[:16] for im in images]


def _answer_schema(q: Question) -> Dict[str, Any]:
    if q.type == "noul":
        props = {"probability_true": {"type": "number"}}
    elif q.type == "choice":
        opts = list(q.criteria)
        props = {
            "choice": {"type": "string", "enum": opts},
            "probabilities": {"type": "object", "additionalProperties": False, "required": opts,
                              "properties": {o: {"type": "number"} for o in opts}},
        }
    elif q.type == "score":
        props = {"level": {"type": "integer"},
                 "probabilities": {"type": "array", "items": {"type": "number"}}}
    else:
        raise ValueError(f"unsupported question type {q.type}")
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def _question_text(q: Question) -> str:
    crit = q.payload()["criteria"]
    if q.type == "noul":
        body = f"True when: {crit['true']}\nFalse when: {crit['false']}"
    elif q.type == "choice":
        body = "\n".join(f"- {k}: {v}" for k, v in crit.items())
    else:
        body = "\n".join(f"{i}: {v}" for i, v in enumerate(crit))
    return f"Question: {q.instructions}\n{body}\nGive probabilities that sum to 1. Say unsure through the probabilities."


def _to_answer(q: Question, data: Optional[Dict[str, Any]], source: str, error: str = "") -> Answer:
    if not data:
        return Answer(q.name, "unassessable", source, error=error or "no answer")
    try:
        if q.type == "noul":
            p = float(data["probability_true"])
            return Answer(q.name, "ok", source, value=p >= 0.5, prob=p, probs={"true": p, "false": 1 - p})
        if q.type == "choice":
            probs = {k: float(v) for k, v in data["probabilities"].items()}
            return Answer(q.name, "ok", source, value=data["choice"], prob=probs.get(data["choice"]), probs=probs)
        probs_list = [float(x) for x in data.get("probabilities", [])]
        level = int(data["level"])
        return Answer(q.name, "ok", source, value=level,
                      prob=probs_list[level] if level < len(probs_list) else None,
                      probs={str(i): v for i, v in enumerate(probs_list)})
    except Exception as e:  # unreadable answer
        return Answer(q.name, "unassessable", source, error=f"unreadable answer: {e}")


def _first(d: Dict[str, Any], *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


def _parse_jev(raw: Dict[str, Any], questions: Sequence[Question]) -> Dict[str, Answer]:
    """Jev answers: {"answers": {name: {"noul": p} | {"choice", "probabilities": {..}} | {"score", ..}}}."""
    answers = _first(raw, "answers") or {}
    out: Dict[str, Answer] = {}
    for q in questions:
        a = answers.get(q.name)
        if a is None:
            out[q.name] = Answer(q.name, "unassessable", "jev", error="question missing from response")
            continue
        try:
            if q.type == "noul":
                p = float(_first(a, "noul", "probability"))
                out[q.name] = Answer(q.name, "ok", "jev", value=p >= 0.5, prob=p, probs={"true": p, "false": 1 - p})
            elif q.type == "choice":
                probs = {k: float(v) for k, v in (_first(a, "probabilities") or {}).items()}
                out[q.name] = Answer(q.name, "ok", "jev", value=a["choice"], prob=probs.get(a["choice"]), probs=probs)
            else:
                out[q.name] = Answer(q.name, "ok", "jev", value=_first(a, "score", "level"),
                                     probs=_first(a, "probabilities"))
        except Exception as e:
            out[q.name] = Answer(q.name, "unassessable", "jev", error=f"unreadable answer: {e}")
    return out


def _parse_openai_decisions(raw: Dict[str, Any], questions: Sequence[Question]) -> Dict[str, Answer]:
    by_name = {a.get("name"): a for a in raw.get("answers", [])}
    out: Dict[str, Answer] = {}
    for q in questions:
        a = by_name.get(q.name)
        if a is None:
            out[q.name] = Answer(q.name, "unassessable", "openai_decisions", error="question missing from response")
            continue
        try:
            if q.type == "noul":
                p = float(a["probability"])
                out[q.name] = Answer(q.name, "ok", "openai_decisions", value=p >= 0.5, prob=p,
                                     probs={"true": p, "false": 1 - p})
            elif q.type == "choice":
                probs = {str(x["value"]): float(x["probability"]) for x in a.get("probabilities", [])}
                out[q.name] = Answer(q.name, "ok", "openai_decisions", value=a["choice"],
                                     prob=probs.get(a["choice"]), probs=probs)
            else:
                probs = {str(x.get("label", x.get("value"))): float(x["probability"]) for x in a.get("probabilities", [])}
                out[q.name] = Answer(q.name, "ok", "openai_decisions", value=float(a["score"]), probs=probs)
        except Exception as e:
            out[q.name] = Answer(q.name, "unassessable", "openai_decisions", error=f"unreadable answer: {e}")
    return out


class SolClient:
    def __init__(self, model: str, effort: str, cache_dir: Path):
        self.model, self.effort = model, effort
        self.cache = DiskCache(Path(cache_dir) / "sol")
        self.last_error = ""

    @property
    def available(self) -> bool:
        if not os.getenv("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def json(self, prompt: str, schema: Dict[str, Any], images: Sequence[ImageLike] = (),
             name: str = "answer") -> Optional[Dict[str, Any]]:
        key = self.cache.key({"m": self.model, "e": self.effort, "p": prompt, "s": schema,
                              "i": _image_ids(images)})
        hit = self.cache.get(key)
        if hit is not None:
            LEDGER.api(self.model, 0.0, 0.0, hit.get("usd", float("nan")), hit.get("tokens_in"),
                       hit.get("tokens_out"), cached=True, seconds_uncached=hit.get("seconds"))
            return hit["data"]
        if not self.available:
            self.last_error = "OPENAI_API_KEY not set"
            return None
        t0 = time.perf_counter()
        try:
            from openai import OpenAI

            content = [{"type": "input_text", "text": prompt}]
            content += [{"type": "input_image", "image_url": data_uri(im)} for im in images]
            resp = OpenAI().responses.create(
                model=self.model,
                reasoning={"effort": self.effort},
                input=[{"role": "user", "content": content}],
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
            )
            data = json.loads(resp.output_text)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            LEDGER.api(self.model, time.perf_counter() - t0, float("nan"), float("nan"))
            return None
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) is not None else {}
        usd = api_usd(self.model, usage)
        seconds = time.perf_counter() - t0
        LEDGER.api(self.model, seconds, usd, usd, usage.get("input_tokens"), usage.get("output_tokens"))
        self.cache.put(key, {"data": data, "usd": usd, "seconds": seconds,
                             "tokens_in": usage.get("input_tokens"), "tokens_out": usage.get("output_tokens")})
        return data

    def ask(self, q: Question, state_text: str = "", images: Sequence[ImageLike] = (),
            source: str = "sol") -> Answer:
        prompt = (state_text + "\n\n" if state_text else "") + _question_text(q)
        data = self.json(prompt, _answer_schema(q), images, name=q.name)
        return _to_answer(q, data, source, self.last_error)


class OpenAIDecisionsClient:
    """Closed questions on text and images. Falls back to Sol on any failure."""

    def __init__(self, model: str, url: str, sol: SolClient, cache_dir: Path):
        self.model, self.url, self.sol = model, url, sol
        self.cache = DiskCache(Path(cache_dir) / "openai_decisions")
        self.last_error = ""

    @property
    def available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY") and self.model)

    def ask(self, questions: Sequence[Question], state_text: str = "",
            images: Sequence[ImageLike] = ()) -> Dict[str, Answer]:
        if images:
            content = [{"type": "input_text", "text": state_text or "Image."}]
            content += [{"type": "input_image", "image_url": data_uri(im)} for im in images]
            inp: Any = [{"role": "user", "content": content}]
        else:
            inp = state_text
        qs = [q.decisions_payload() for q in questions]
        key = self.cache.key({"m": self.model, "t": state_text, "q": qs, "i": _image_ids(images)})
        hit = self.cache.get(key)
        raw = None
        if hit is not None:
            raw = hit["raw"]
            LEDGER.api(self.model, 0.0, 0.0, hit.get("usd", float("nan")), hit.get("tokens_in"), 0, cached=True,
                       seconds_uncached=hit.get("seconds"))
        elif self.available:
            t0 = time.perf_counter()
            try:
                import requests

                r = requests.post(self.url, json={"model": self.model, "input": inp, "questions": qs},
                                  headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, timeout=90)
                r.raise_for_status()
                raw = r.json()
                usage = raw.get("usage") or {}
                usd = api_usd(self.model, usage)
                seconds = time.perf_counter() - t0
                LEDGER.api(self.model, seconds, usd, usd, usage.get("input_tokens"), usage.get("output_tokens"))
                self.cache.put(key, {"raw": raw, "usd": usd, "seconds": seconds,
                                     "tokens_in": usage.get("input_tokens")})
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                LEDGER.api(self.model, time.perf_counter() - t0, float("nan"), float("nan"))
                raw = None
        if raw is not None:
            out = _parse_openai_decisions(raw, questions)
            if all(a.status == "ok" for a in out.values()):
                return out
        return {q.name: self.sol.ask(q, state_text, images, source="sol_fallback") for q in questions}


class JevClient:
    """TypeSafe's own API when JEV_API_KEY is set, else OpenRouter with OPENROUTER_API_KEY."""

    def __init__(self, model: str, url: str, cache_dir: Path, openrouter_model: str = "",
                 openrouter_url: str = ""):
        if not os.getenv("JEV_API_KEY") and os.getenv("OPENROUTER_API_KEY") and openrouter_url:
            model, url = openrouter_model, openrouter_url
            self.key_name = "OPENROUTER_API_KEY"
        else:
            self.key_name = "JEV_API_KEY"
        self.model, self.url = model, url
        self.cache = DiskCache(Path(cache_dir) / "jev")
        self.last_error = ""
        self.resolved_model = ""

    @property
    def available(self) -> bool:
        return bool(os.getenv(self.key_name))

    def ask(self, questions: Sequence[Question], state: Any) -> Dict[str, Answer]:
        body = {"model": self.model, "state": state,
                "questions": {q.name: q.payload() for q in questions}}
        key = self.cache.key(body)
        hit = self.cache.get(key)
        if hit is not None:
            raw = hit["raw"]
            LEDGER.api(self.model, 0.0, 0.0, hit.get("usd", float("nan")), hit.get("tokens_in"), 0, cached=True,
                       seconds_uncached=hit.get("seconds"))
        else:
            if not self.available:
                return {q.name: Answer(q.name, "unassessable", "jev", error=f"{self.key_name} not set")
                        for q in questions}
            t0 = time.perf_counter()
            try:
                import requests

                r = requests.post(self.url, json=body, timeout=90, headers={
                    "Authorization": f"Bearer {os.environ[self.key_name]}",
                    "Content-Type": "application/json"})
                r.raise_for_status()
                raw = r.json()
                usage = raw.get("usage") or {}
                usd = float(usage["cost"]) if "cost" in usage else api_usd("jev", usage)
                seconds = time.perf_counter() - t0
                LEDGER.api(self.model, seconds, usd, usd, usage.get("input_tokens"), usage.get("output_tokens"))
                self.cache.put(key, {"raw": raw, "usd": usd, "seconds": seconds,
                                     "tokens_in": usage.get("input_tokens")})
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                LEDGER.api(self.model, time.perf_counter() - t0, float("nan"), float("nan"))
                return {q.name: Answer(q.name, "unassessable", "jev", error=self.last_error)
                        for q in questions}
        self.resolved_model = raw.get("model", self.resolved_model)
        return _parse_jev(raw, questions)


def build_clients(cfg) -> SimpleNamespace:
    cache = Path(cfg.cache_dir) / "api"
    sol = SolClient(cfg.sol_model, cfg.sol_reasoning_effort, cache)
    decisions = OpenAIDecisionsClient(cfg.decisions_model, cfg.decisions_url, sol, cache)
    jev = JevClient(cfg.jev_model, cfg.jev_url, cache, cfg.jev_openrouter_model, cfg.jev_openrouter_url)
    return SimpleNamespace(sol=sol, decisions=decisions, jev=jev)


def client_status(clients: SimpleNamespace) -> List[Dict[str, Any]]:
    return [
        {"client": "Sol", "model": clients.sol.model, "role": "Questions on a set of frames. Parser fallback.",
         "ready": clients.sol.available},
        {"client": "OpenAI Decisions", "model": clients.decisions.model,
         "role": "Closed questions on one image. Falls back to Sol.", "ready": clients.decisions.available},
        {"client": "Jev", "model": clients.jev.model, "role": "Closed questions on caption text.",
         "ready": clients.jev.available},
    ]
