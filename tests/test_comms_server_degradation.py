"""The comms MCP server must work on the default install.

Core tools (modality id, information theory) run with no heavy deps, and every
optional-library tool returns an actionable enable-message instead of raising
when its library is absent. Loaded in isolation like test_mcp_common_truncate.
"""
import asyncio
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "mcp_servers" / "comms_server.py"


def _load():
    loader = importlib.machinery.SourceFileLoader("odysseus_comms_server", str(_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _call(module, name, args):
    return asyncio.run(module.call_tool(name, args))[0].text


def test_lists_all_tools():
    m = _load()
    names = {t.name for t in asyncio.run(m.list_tools())}
    assert {"comm_identify_modality", "comm_infotheory", "comm_discover",
            "comm_language", "comm_signal", "comm_quantum", "comm_embed",
            "comm_bioacoustic"} <= names


def test_core_identify_and_infotheory_work_without_heavy_deps():
    m = _load()
    out = _call(m, "comm_identify_modality", {"signal": "0100100001101001"})
    assert "binary" in out
    out = _call(m, "comm_infotheory", {"action": "entropy", "data": "aaaabbbb"})
    assert '"entropy_bits_per_symbol": 1.0' in out
    out = _call(m, "comm_infotheory",
                {"action": "channel_established", "sent": "abcabc", "received": "abcabc"})
    assert '"established": true' in out


def test_optional_tool_returns_enable_message_when_lib_absent():
    m = _load()
    # transformers/imagebind are not in the default install -> graceful message.
    out = _call(m, "comm_language", {"action": "translate", "text": "hola"})
    assert "requires the optional library" in out and "pip install" in out
    out = _call(m, "comm_embed", {"action": "embed", "text": "x"})
    assert "pip install" in out


def test_quantum_describe_protocol_is_always_available():
    m = _load()
    out = _call(m, "comm_quantum", {"action": "describe_protocol", "protocol": "bb84"})
    assert "BB84" in out


def test_discover_without_llm_returns_clear_error():
    m = _load()
    # No model configured in the test DB -> actionable message, not a crash.
    out = _call(m, "comm_discover", {"signal": "aGVsbG8="})
    assert "LLM endpoint" in out or "established" in out


def test_unknown_tool_name():
    m = _load()
    out = _call(m, "comm_nonexistent", {})
    assert "Unknown tool" in out
