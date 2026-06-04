# src/protocol_discovery.py
"""
Protocol-discovery engine for the Universal Communication Agent.

A structural sibling of src/deep_research.py's DeepResearcher: an iterative
plan → propose → attempt → measure → STOP loop. The decisive difference is the
**stop condition**. Deep Research asks an LLM "is this report good enough?".
Here the loop stops on an *objective information-theory metric* computed by
src/comms_infotheory.py: a channel is established when the receiver recovers at
least ``mi_threshold`` of the sent signal's information (normalized mutual
information). The LLM proposes *what to try*; mutual information decides *whether
it worked* — so the agent cannot hallucinate success.

When known protocols are exhausted (``max_empty_rounds`` with no improvement),
the loop escalates into **invent mode**: it asks the model to design a novel
encoding from first principles (à la Lincos/CosmicOS), then measures the
invention with the same metric. The discrete substitution case is executable
today; richer inventions (multi-agent emergent communication via EGG) plug in as
an optional ``attempt_fn`` in a later milestone.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Awaitable, Callable, Dict, List, Optional

from src.research_utils import strip_thinking

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PROTOCOL_PLAN_PROMPT = """\
You are a communication-theory strategist. An unknown signal has been received \
and we must establish communication: recover the intended message, or determine \
how to talk back.

**Signal description / context:** {context}

**Information-theoretic profile of the signal:**
{profile}

Analyze before acting:
1. What modality could this be (text/cipher, binary, optical/RF waveform, \
quantum-state description, bioacoustic, mathematical, or unknown)?
2. What does the entropy/redundancy suggest about whether it is an intentional, \
structured message versus noise?
3. What does success look like — what would a recovered channel produce?

Return a JSON object:
- "modality_hypotheses": array of 2-4 likely modalities, most likely first
- "reasoning": one or two sentences on the entropy/structure evidence
- "success_criteria": one sentence describing a recovered channel
"""

PROTOCOL_GEN_PROMPT = """\
You are proposing communication methods to ATTEMPT this round.

**Signal context:** {context}
**Signal profile:** {profile}
**Plan:** {plan}
**Best result so far:** {best}
**Already tried (do not repeat):** {tried}
**Round:** {round_num}

{mode_instruction}

Return ONLY a JSON array of method objects. Each object:
{{"method": "<short name>", "rationale": "<why this could work>"}}
{invent_schema}
Example: [{{"method": "base64", "rationale": "alphabet looks like base64"}}]
"""

_KNOWN_MODE = """\
Propose {n} concrete, KNOWN decoding/communication methods to try, best-bet \
first. Recognized method names include: identity/ascii, hex, base64, binary, \
morse, rot13, caesar shift N, atbash, reverse. Pick the ones the signal profile \
most supports."""

_INVENT_MODE = """\
Known methods have not established a channel. Switch to INVENT mode: design a \
NOVEL decoding scheme from first principles. For a substitution-style invention, \
include a "mapping" object of single-character -> single-character replacements \
that you hypothesize will reveal a message (e.g. from frequency analysis). \
Propose {n} candidate inventions, each with a clear rationale."""

_INVENT_SCHEMA = """\
For invented substitution ciphers, also include: "mapping": {"a": "x", ...} \
(single chars). Inventions without a mapping are recorded as proposed protocols."""

PROTOCOL_SYNTHESIZE_PROMPT = """\
You maintain a "communication dossier" tracking the effort to establish a channel.

**Signal context:** {context}
**Current dossier:**
{dossier}

**This round's attempts (method → measured normalized mutual information 0..1):**
{attempts}

Update the dossier. State the best method and its measured score so far, what \
has been ruled out, and what to try next. Be concise and factual — never claim a \
channel is established unless an attempt's measured score met the threshold. \
Write only the updated dossier.
"""


class ProtocolDiscoverer:
    """Iterative, metric-driven search for a working communication method."""

    def __init__(
        self,
        llm_endpoint: str,
        llm_model: str,
        llm_headers: Optional[Dict] = None,
        max_rounds: int = 6,
        max_time: int = 240,
        min_rounds: int = 1,
        max_empty_rounds: int = 2,
        mi_threshold: float = 0.5,
        candidates_per_round: int = 4,
        progress_callback: Optional[Callable] = None,
        attempt_fn: Optional[Callable[[Dict, str], object]] = None,
    ):
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.llm_headers = llm_headers
        self.max_rounds = max_rounds
        self.max_time = max_time
        self.min_rounds = min_rounds
        self.max_empty_rounds = max_empty_rounds
        self.mi_threshold = mi_threshold
        self.candidates_per_round = candidates_per_round
        self._progress = progress_callback
        # attempt_fn(method, signal) -> attempt dict (may be sync or async).
        # Default: the built-in codec library + communicativeness scorer.
        if attempt_fn is None:
            from src.comms_codecs import builtin_attempt
            attempt_fn = builtin_attempt
        self._attempt_fn = attempt_fn

        self._cancelled = False
        self._start_time: float = 0.0
        self.round_count = 0
        self.tried: List[str] = []
        self.attempts_log: List[Dict] = []
        self.best: Dict = {"method": None, "score": 0.0, "established": False, "decoded": ""}
        self.dossier: str = ""

    def cancel(self):
        self._cancelled = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def discover(self, signal: str, context: str = "") -> Dict:
        """Run the discovery loop. Returns a dossier dict with the outcome."""
        from src.comms_codecs import signal_profile

        self._start_time = time.time()
        profile = signal_profile(signal)
        profile_str = json.dumps(profile, indent=2)

        self._emit(phase="planning")
        plan = await self._plan(context or "(no description provided)", profile_str)

        no_improve = 0
        for round_num in range(1, self.max_rounds + 1):
            self.round_count = round_num
            if self._cancelled or self._time_exceeded():
                break

            invent = no_improve >= self.max_empty_rounds
            mode = "invent" if invent else "known"
            self._emit(phase="proposing", round=round_num, mode=mode)

            candidates = await self._propose(
                context, profile_str, plan, round_num, invent
            )
            if not candidates:
                logger.info("Round %d: no candidate methods proposed", round_num)
                no_improve += 1
                if no_improve >= self.max_empty_rounds and round_num >= self.min_rounds:
                    # Already in invent mode and still nothing — give up gracefully.
                    if invent:
                        break
                continue

            prev_best = self.best["score"]
            round_attempts: List[Dict] = []
            for cand in candidates:
                if self._cancelled or self._time_exceeded():
                    break
                name = cand.get("method") or cand.get("name") or str(cand)
                self.tried.append(name)
                self._emit(phase="attempting", round=round_num, method=name, mode=mode)
                attempt = await self._run_attempt(cand, signal)
                metrics = self._measure(attempt)
                record = {
                    "round": round_num,
                    "mode": mode,
                    "method": name,
                    "rationale": cand.get("rationale", ""),
                    **metrics,
                    "decoded": attempt.get("decoded", ""),
                    "note": attempt.get("note", ""),
                    "invented": attempt.get("invented", False),
                }
                self.attempts_log.append(record)
                round_attempts.append(record)
                self._emit(
                    phase="measured",
                    round=round_num,
                    method=name,
                    nmi=metrics["normalized_mutual_information"],
                    established=metrics["established"],
                )
                if metrics["normalized_mutual_information"] > self.best["score"]:
                    self.best = {
                        "method": name,
                        "score": metrics["normalized_mutual_information"],
                        "established": metrics["established"],
                        "decoded": attempt.get("decoded", ""),
                        "metrics": metrics,
                        "invented": attempt.get("invented", False),
                    }
                # METRIC-DRIVEN STOP: success the moment the threshold is crossed.
                if metrics["established"] and round_num >= self.min_rounds:
                    self.dossier = await self._synthesize(context, round_attempts)
                    self._emit(phase="established", method=name,
                               nmi=metrics["normalized_mutual_information"])
                    return self._result(established=True, signal_profile=profile)

            # SYNTHESIZE dossier for this round.
            self.dossier = await self._synthesize(context, round_attempts)
            if self.best["score"] > prev_best + 1e-6:
                no_improve = 0
            else:
                no_improve += 1

        self._emit(phase="exhausted", best_method=self.best["method"],
                   best_nmi=self.best["score"])
        return self._result(established=self.best["established"], signal_profile=profile)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    async def _plan(self, context: str, profile_str: str) -> str:
        prompt = PROTOCOL_PLAN_PROMPT.format(context=context, profile=profile_str)
        try:
            resp = await self._llm([{"role": "user", "content": prompt}],
                                   temperature=0.3, max_tokens=700, timeout=40)
            parsed = self._parse_json_object(resp)
            if parsed:
                parts = []
                if parsed.get("modality_hypotheses"):
                    parts.append("Modality hypotheses: "
                                 + ", ".join(str(m) for m in parsed["modality_hypotheses"]))
                if parsed.get("reasoning"):
                    parts.append("Reasoning: " + str(parsed["reasoning"]))
                if parsed.get("success_criteria"):
                    parts.append("Success: " + str(parsed["success_criteria"]))
                return "\n".join(parts) if parts else resp
            return resp
        except Exception as e:
            logger.warning("Protocol planning failed: %s", e)
            return ""

    async def _propose(self, context: str, profile_str: str, plan: str,
                       round_num: int, invent: bool) -> List[Dict]:
        n = self.candidates_per_round
        mode_instruction = (_INVENT_MODE if invent else _KNOWN_MODE).format(n=n)
        prompt = PROTOCOL_GEN_PROMPT.format(
            context=context or "(none)",
            profile=profile_str,
            plan=plan or "(no plan)",
            best=f"{self.best['method']} (score {self.best['score']:.2f})"
            if self.best["method"] else "(nothing tried yet)",
            tried=", ".join(self.tried[-20:]) or "(none)",
            round_num=round_num,
            mode_instruction=mode_instruction,
            invent_schema=_INVENT_SCHEMA if invent else "",
        )
        try:
            resp = await self._llm([{"role": "user", "content": prompt}],
                                   temperature=0.6, max_tokens=1200, timeout=60)
            objs = self._parse_json_objects(resp)
            # Drop methods we've already tried by name (keep inventions w/ mappings).
            fresh = []
            for o in objs:
                nm = (o.get("method") or o.get("name") or "").strip()
                if o.get("mapping") or nm.lower() not in {t.lower() for t in self.tried}:
                    fresh.append(o)
            return fresh[:n]
        except Exception as e:
            logger.warning("Proposal generation failed: %s", e)
            return []

    async def _run_attempt(self, method: Dict, signal: str) -> Dict:
        try:
            res = self._attempt_fn(method, signal)
            if hasattr(res, "__await__"):
                res = await res
            if not isinstance(res, dict):
                return {"method": str(method), "ok": False, "score": 0.0,
                        "note": "attempt_fn returned non-dict"}
            return res
        except Exception as e:
            logger.warning("Attempt failed for %s: %s", method, e)
            return {"method": str(method), "ok": False, "score": 0.0,
                    "note": f"attempt raised: {e}"}

    def _measure(self, attempt: Dict) -> Dict:
        """Turn an attempt result into objective channel metrics.

        Two contracts are supported so the engine stays universal:
        - aligned symbol sequences (``sent`` & ``received``) -> full info-theory
          measurement via comms_infotheory.channel_established (used by real
          channels, tests, RF/quantum/EGG attempts);
        - a precomputed ``score`` 0..1 (used by the built-in codec evaluator).
        """
        from src.comms_infotheory import channel_established

        if attempt.get("sent") is not None and attempt.get("received") is not None:
            m = channel_established(attempt["sent"], attempt["received"],
                                    threshold=self.mi_threshold)
            return {
                "normalized_mutual_information": m["normalized_mutual_information"],
                "mutual_information_bits": m["mutual_information_bits"],
                "channel_capacity_bits": m["channel_capacity_bits"],
                "established": m["established"],
                "reason": m["reason"],
            }
        score = float(attempt.get("score", 0.0) or 0.0)
        score = max(0.0, min(1.0, score))
        return {
            "normalized_mutual_information": round(score, 4),
            "mutual_information_bits": None,
            "channel_capacity_bits": None,
            "established": bool(attempt.get("ok") and score >= self.mi_threshold),
            "reason": attempt.get("note", ""),
        }

    async def _synthesize(self, context: str, round_attempts: List[Dict]) -> str:
        lines = [
            f"- {a['method']}: NMI={a['normalized_mutual_information']:.2f} "
            f"{'ESTABLISHED' if a['established'] else ''} {a.get('note', '')}".rstrip()
            for a in round_attempts
        ]
        prompt = PROTOCOL_SYNTHESIZE_PROMPT.format(
            context=context or "(none)",
            dossier=self.dossier or "(first round — no dossier yet)",
            attempts="\n".join(lines) or "(no attempts)",
        )
        try:
            return await self._llm([{"role": "user", "content": prompt}],
                                   temperature=0.3, max_tokens=1200, timeout=90)
        except Exception as e:
            logger.warning("Dossier synthesis failed: %s", e)
            return self.dossier  # keep prior dossier on failure

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------
    def _result(self, established: bool, signal_profile: Dict) -> Dict:
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        return {
            "established": established,
            "best_method": self.best["method"],
            "best_normalized_mutual_information": self.best["score"],
            "best_decoded": self.best.get("decoded", ""),
            "best_invented": self.best.get("invented", False),
            "threshold": self.mi_threshold,
            "rounds": self.round_count,
            "attempts": self.attempts_log,
            "signal_profile": signal_profile,
            "dossier": self.dossier,
            "elapsed_seconds": round(elapsed, 2),
        }

    # ------------------------------------------------------------------
    # LLM + parsing helpers (mirrors deep_research.py, kept self-contained)
    # ------------------------------------------------------------------
    async def _llm(self, messages: List[Dict], temperature: float = 0.3,
                   max_tokens: int = 1024, timeout: int = 60) -> str:
        from src.llm_core import llm_call_async
        response = await llm_call_async(
            url=self.llm_endpoint,
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            headers=self.llm_headers,
            timeout=timeout,
        )
        return strip_thinking(response)

    def _emit(self, **kwargs):
        if self._progress:
            try:
                self._progress(kwargs)
            except Exception:
                pass

    def _time_exceeded(self) -> bool:
        return (time.time() - self._start_time) > self.max_time

    @staticmethod
    def _strip_code_block(text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _parse_json_object(self, text: str) -> Optional[Dict]:
        text = self._strip_code_block(text)
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                obj = json.loads(match.group())
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                pass
        return None

    def _parse_json_objects(self, text: str) -> List[Dict]:
        """Extract a JSON array of method objects from LLM output, robustly.

        Mirrors deep_research's array-repair philosophy: tolerate code fences,
        echoed examples, and trailing prose. Keeps the LAST parseable array.
        """
        text = self._strip_code_block(text)
        # Direct parse.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [o for o in parsed if isinstance(o, dict)]
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass
        # Last parseable [...] array (the model often echoes the Example first).
        last = None
        for m in re.finditer(r"\[[\s\S]*?\]", text):
            try:
                cand = json.loads(m.group())
                if isinstance(cand, list) and any(isinstance(o, dict) for o in cand):
                    last = cand
            except json.JSONDecodeError:
                continue
        if last is not None:
            return [o for o in last if isinstance(o, dict)]
        # Greedy outermost array.
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                cand = json.loads(match.group())
                if isinstance(cand, list):
                    return [o for o in cand if isinstance(o, dict)]
            except json.JSONDecodeError:
                pass
        # Harvest standalone {...} objects as a last resort.
        objs = []
        for m in re.finditer(r"\{[^{}]*\}", text):
            try:
                o = json.loads(m.group())
                if isinstance(o, dict) and (o.get("method") or o.get("name")):
                    objs.append(o)
            except json.JSONDecodeError:
                continue
        return objs
