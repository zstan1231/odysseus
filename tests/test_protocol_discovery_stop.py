"""Discovery loop: metric-driven stop and invent-mode escalation.

The engine must stop the moment an attempt's measured mutual information crosses
the threshold, and must escalate from known methods to invent mode when known
methods stall — all driven by objective metrics, with the LLM stubbed out.
"""
import pytest

from src.protocol_discovery import ProtocolDiscoverer


def _make_llm(proposal_rounds):
    """Stub LLM: returns canned proposals per call, JSON-encoded.

    ``proposal_rounds`` is a list; each element is the JSON array string to
    return for that proposal call (last element repeats). Planning/synthesis
    calls return trivial text.
    """
    import json

    state = {"i": 0}

    async def _llm(messages, **kw):
        content = messages[0]["content"]
        if "communication-theory strategist" in content:
            return json.dumps({"modality_hypotheses": ["cipher"],
                               "reasoning": "x", "success_criteria": "y"})
        if "proposing communication methods" in content:
            # Invent mode is detectable from the instruction text.
            if "INVENT mode" in content:
                return json.dumps([{"method": "inventX", "rationale": "novel"}])
            i = state["i"]
            state["i"] += 1
            return proposal_rounds[min(i, len(proposal_rounds) - 1)]
        return "dossier text"

    return _llm


_SENT = [i % 4 for i in range(400)]  # 4-symbol source, H(X) = 2 bits


def _aligned_attempt(method, signal):
    """Fake channel: 'good'/'inventX' establish; everything else is noise.

    Uses long sequences so the measured mutual information is statistically
    meaningful: the establishing methods return a perfect copy (NMI=1); noise
    methods return a constant (NMI=0).
    """
    name = method.get("method", "")
    if name in ("good", "inventX"):
        return {"method": name, "ok": True, "sent": list(_SENT),
                "received": list(_SENT), "invented": name == "inventX"}
    return {"method": name, "ok": True, "sent": list(_SENT),
            "received": [0] * len(_SENT)}


@pytest.mark.asyncio
async def test_stops_when_mi_crosses_threshold():
    import json
    d = ProtocolDiscoverer("http://x", "m", min_rounds=1, max_rounds=6,
                           candidates_per_round=1, attempt_fn=_aligned_attempt)
    d._llm = _make_llm([json.dumps([{"method": "bad"}]),
                        json.dumps([{"method": "good"}])])
    res = await d.discover("some signal", context="test")
    assert res["established"] is True
    assert res["best_method"] == "good"
    assert res["best_normalized_mutual_information"] > 0.99
    assert res["rounds"] == 2  # stopped as soon as 'good' established


@pytest.mark.asyncio
async def test_escalates_to_invent_mode_when_known_methods_stall():
    import json
    d = ProtocolDiscoverer("http://x", "m", min_rounds=1, max_rounds=6,
                           max_empty_rounds=1, candidates_per_round=1,
                           attempt_fn=_aligned_attempt)
    # Every known proposal is 'bad' (never establishes) -> must escalate.
    d._llm = _make_llm([json.dumps([{"method": "bad"}])])
    res = await d.discover("some signal", context="test")
    modes = [a["mode"] for a in res["attempts"]]
    assert "invent" in modes
    assert res["established"] is True
    assert res["best_method"] == "inventX"
    assert res["best_invented"] is True


@pytest.mark.asyncio
async def test_no_method_works_returns_unestablished():
    import json
    def always_noise(method, signal):
        return {"method": method.get("method", ""), "ok": True,
                "sent": list(_SENT), "received": [0] * len(_SENT)}

    d = ProtocolDiscoverer("http://x", "m", min_rounds=1, max_rounds=3,
                           max_empty_rounds=1, candidates_per_round=1,
                           attempt_fn=always_noise)
    d._llm = _make_llm([json.dumps([{"method": "bad"}])])
    res = await d.discover("noise", context="test")
    assert res["established"] is False
    assert res["best_normalized_mutual_information"] < 0.5


def test_measure_handles_both_contracts():
    d = ProtocolDiscoverer("http://x", "m")
    # Aligned sequences -> full info-theory measurement.
    aligned = d._measure({"sent": list("abcabc"), "received": list("abcabc")})
    assert aligned["established"] is True
    assert aligned["mutual_information_bits"] is not None
    # Precomputed score (codec evaluator contract).
    scored = d._measure({"ok": True, "score": 0.8})
    assert scored["established"] is True
    assert scored["normalized_mutual_information"] == 0.8
    low = d._measure({"ok": True, "score": 0.2})
    assert low["established"] is False
