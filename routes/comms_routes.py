"""Universal Communication Agent routes — /api/comms/*.

Backs the dedicated Comms tab. The headline endpoint runs the iterative
protocol-discovery loop (src/protocol_discovery.py) as a background task and
streams its progress over SSE, mirroring the Deep Research panel. The one-shot
endpoints expose the always-available core tools (modality identification and
information-theoretic measurement) directly.
"""
import asyncio
import json
import logging
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.auth_helpers import _auth_disabled, get_current_user

logger = logging.getLogger(__name__)

# In-memory job registry: job_id -> job dict. Discovery runs are short-lived and
# owner-scoped; nothing here needs to survive a restart.
_JOBS: Dict[str, Dict] = {}
_MAX_JOBS = 200


class DiscoverRequest(BaseModel):
    signal: str = Field(..., description="The signal/payload to establish communication through.")
    context: str = ""
    threshold: float = 0.5
    max_rounds: int = 6
    max_time: int = 180


class IdentifyRequest(BaseModel):
    signal: str


class MeasureRequest(BaseModel):
    action: str = "channel_established"
    data: Optional[str] = None
    sent: Optional[str] = None
    received: Optional[str] = None
    threshold: float = 0.5


def _resolve_comms_endpoint():
    """Resolve (url, model, headers) for the discovery LLM, like Deep Research."""
    from src.endpoint_resolver import resolve_endpoint
    for role in ("research", "utility", "default", "chat"):
        url, model, headers = resolve_endpoint(role)
        if url and model:
            return url, model, headers
    return None, None, None


def setup_comms_routes(session_manager=None) -> APIRouter:
    router = APIRouter(tags=["comms"])

    def _require_user(request: Request) -> str:
        user = get_current_user(request)
        if not user:
            if _auth_disabled():
                return ""
            raise HTTPException(401, "Not authenticated")
        return user

    def _owns(job_id: str, user: str) -> Dict:
        job = _JOBS.get(job_id)
        if not job or (job.get("owner") or "") != (user or ""):
            raise HTTPException(404, "No discovery job found")
        return job

    # -- core one-shot tools (no LLM, no heavy deps) ------------------------
    @router.post("/api/comms/identify")
    async def comms_identify(body: IdentifyRequest, request: Request):
        _require_user(request)
        from src.comms_codecs import signal_profile
        if not body.signal:
            raise HTTPException(400, "signal is required")
        return signal_profile(body.signal)

    @router.post("/api/comms/measure")
    async def comms_measure(body: MeasureRequest, request: Request):
        _require_user(request)
        import src.comms_infotheory as it
        if body.action == "entropy":
            data = it.symbolize(body.data or "")
            return {
                "action": "entropy",
                "entropy_bits_per_symbol": round(it.entropy_of_sequence(data), 4),
                "redundancy": round(it.estimate_redundancy(data), 4),
                "alphabet_size": len(set(data)),
                "length": len(data),
            }
        sent = it.symbolize(body.sent or "")
        received = it.symbolize(body.received or "")
        if not sent or not received:
            raise HTTPException(400, "sent and received are required")
        if body.action == "mutual_information":
            return {
                "mutual_information_bits": round(it.mutual_information(sent, received), 4),
                "normalized_mutual_information": round(
                    it.normalized_mutual_information(sent, received), 4),
            }
        if body.action == "channel_capacity":
            return {"channel_capacity_bits_per_use": round(it.channel_capacity(sent, received), 4)}
        return it.channel_established(sent, received, threshold=body.threshold)

    # -- discovery loop (background + SSE) ----------------------------------
    @router.post("/api/comms/discover")
    async def comms_discover(body: DiscoverRequest, request: Request):
        user = _require_user(request)
        if not body.signal:
            raise HTTPException(400, "signal is required")
        url, model, headers = _resolve_comms_endpoint()
        if not url or not model:
            raise HTTPException(400, "No chat model configured. Add one in Settings first.")

        job_id = f"comms-{uuid.uuid4().hex[:12]}"
        job = {"owner": user, "status": "running", "events": [], "result": None,
               "discoverer": None, "task": None}
        # Bound the registry so long-lived servers don't leak completed jobs.
        if len(_JOBS) >= _MAX_JOBS:
            for stale in [k for k, v in list(_JOBS.items()) if v.get("status") != "running"][:50]:
                _JOBS.pop(stale, None)
        _JOBS[job_id] = job

        from src.protocol_discovery import ProtocolDiscoverer

        def _on_progress(event: Dict):
            job["events"].append(event)

        disc = ProtocolDiscoverer(
            llm_endpoint=url, llm_model=model, llm_headers=headers,
            max_rounds=max(1, min(20, body.max_rounds)),
            max_time=max(15, min(900, body.max_time)),
            mi_threshold=body.threshold,
            progress_callback=_on_progress,
        )
        job["discoverer"] = disc

        async def _run():
            try:
                result = await disc.discover(body.signal, context=body.context)
                job["result"] = result
                job["status"] = "done"
            except Exception as e:
                logger.exception("Discovery job %s failed", job_id)
                job["result"] = {"error": str(e)[:500]}
                job["status"] = "error"

        job["task"] = asyncio.create_task(_run())
        return {"job_id": job_id, "status": "running"}

    @router.get("/api/comms/stream/{job_id}")
    async def comms_stream(job_id: str, request: Request):
        user = _require_user(request)
        job = _owns(job_id, user)

        async def _generate():
            cursor = 0
            while True:
                events = job["events"]
                while cursor < len(events):
                    yield f"data: {json.dumps(events[cursor])}\n\n"
                    cursor += 1
                if job["status"] != "running":
                    final = {"phase": "complete", "status": job["status"],
                             "final": True, "result": job.get("result")}
                    yield f"data: {json.dumps(final, default=str)}\n\n"
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/api/comms/result/{job_id}")
    async def comms_result(job_id: str, request: Request):
        user = _require_user(request)
        job = _owns(job_id, user)
        return {"status": job["status"], "result": job.get("result")}

    @router.post("/api/comms/cancel/{job_id}")
    async def comms_cancel(job_id: str, request: Request):
        user = _require_user(request)
        job = _owns(job_id, user)
        disc = job.get("discoverer")
        if disc:
            disc.cancel()
        return {"status": "cancelling"}

    return router
