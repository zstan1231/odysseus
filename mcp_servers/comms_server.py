"""
comms_server.py

MCP server for the Universal Communication Agent. Exposes tools to identify a
signal's modality, measure information-theoretic channel metrics, decode/translate
across modalities, and — the centerpiece — run an iterative protocol-discovery
loop that invents a new communication method when no known one works.

Design contract (matches the rest of mcp_servers/): the always-available core
tools (modality identification, information theory, discovery over the built-in
codec library) are pure-Python + numpy and never fail to import. Every heavy
modality decoder (language/optical/quantum/cross-modal/bioacoustic) is imported
*inside* the call, so a missing library yields an actionable "enable it like
this" message instead of crashing the server at startup.
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_servers._common import truncate  # noqa: E402

server = Server("comms")


# ---------------------------------------------------------------------------
# Lazy-import helper for heavy/optional modality libraries.
# ---------------------------------------------------------------------------
def _try_import(modname: str, feature: str, pip_line: str):
    """Return (module, None) or (None, enable_message)."""
    try:
        mod = __import__(modname)
        return mod, None
    except Exception as e:  # ImportError, or a half-installed lib raising on import
        return None, (
            f"{feature} requires the optional library `{modname}`, which is not "
            f"available ({type(e).__name__}: {e}).\n"
            f"Enable it with:  pip install {pip_line}\n"
            f"(declared in requirements-comms.txt; imported lazily so the rest of "
            f"the communication tools keep working without it)."
        )


def _resolve_llm():
    """Resolve an (endpoint, model, headers) triple for the discovery loop.

    Reuses the app's endpoint resolver (DB-backed, works from this subprocess).
    Tries the research role first, then the default chat model.
    """
    try:
        from src.endpoint_resolver import resolve_endpoint
    except Exception:
        return None, None, None
    for role in ("research", "default", "utility"):
        url, model, headers = resolve_endpoint(role)
        if url and model:
            return url, model, headers
    return None, None, None


def _symbols(value):
    """Coerce a tool argument into a symbol list for info-theory measures."""
    from src.comms_infotheory import symbolize
    if isinstance(value, list):
        return value
    return symbolize(value if value is not None else "")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="comm_identify_modality",
            description=(
                "Identify the likely modality of an unknown signal (text/cipher, "
                "binary, hex, morse, optical/RF waveform, quantum, bioacoustic, or "
                "noise) and return its information-theoretic profile (entropy, "
                "redundancy, alphabet). Always available, no heavy deps."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "signal": {"type": "string", "description": "The raw signal as text."},
                },
                "required": ["signal"],
            },
        ),
        Tool(
            name="comm_infotheory",
            description=(
                "Measure information-theoretic channel metrics — the objective "
                "test of whether communication succeeded. Actions: entropy (of "
                "'data'); mutual_information, channel_capacity, channel_established "
                "(between aligned 'sent' and 'received')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["entropy", "mutual_information",
                                 "channel_capacity", "channel_established"],
                    },
                    "data": {"type": "string", "description": "Sequence for entropy."},
                    "sent": {"type": "string", "description": "Transmitted sequence."},
                    "received": {"type": "string", "description": "Observed sequence."},
                    "threshold": {"type": "number",
                                  "description": "NMI threshold for channel_established (default 0.5)."},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="comm_discover",
            description=(
                "Run the iterative protocol-discovery loop on an unknown signal: "
                "propose methods, decode, MEASURE recovered information, and if no "
                "known method works, INVENT a new one — stopping only when mutual "
                "information crosses the threshold. The core 'find a new way to "
                "communicate' capability."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "signal": {"type": "string", "description": "The signal to decode."},
                    "context": {"type": "string",
                                "description": "What you know about the source/counterparty."},
                    "threshold": {"type": "number",
                                  "description": "NMI success threshold (default 0.5)."},
                    "max_rounds": {"type": "integer",
                                   "description": "Discovery rounds (default 3 for tool calls)."},
                },
                "required": ["signal"],
            },
        ),
        Tool(
            name="comm_language",
            description=(
                "Cross-language communication. Actions: detect_language, translate, "
                "transcribe (speech). Uses SeamlessM4T/NLLB/Whisper when installed; "
                "otherwise returns enable instructions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["detect_language", "translate", "transcribe"]},
                    "text": {"type": "string"},
                    "audio_path": {"type": "string"},
                    "source_lang": {"type": "string"},
                    "target_lang": {"type": "string"},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="comm_signal",
            description=(
                "Optical/RF physical-layer analysis. Actions: analyze, demodulate, "
                "decode_fec. Uses scikit-dsp-comm / Sionna when installed; otherwise "
                "returns enable instructions plus an analytic description."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["analyze", "demodulate", "decode_fec"]},
                    "data": {"type": "string"},
                    "scheme": {"type": "string",
                               "description": "Modulation scheme, e.g. BPSK/QPSK/OOK."},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="comm_quantum",
            description=(
                "Quantum communication. Actions: describe_protocol (BB84, "
                "teleportation, superdense — always available), simulate_channel "
                "(uses SQUANCH/QuNetSim when installed)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["describe_protocol", "simulate_channel"]},
                    "protocol": {"type": "string",
                                 "description": "e.g. bb84, teleportation, superdense"},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="comm_embed",
            description=(
                "Cross-modal (any-to-any) embedding and similarity using ImageBind "
                "when installed; otherwise returns enable instructions. Binds image/"
                "text/audio into one space to align modalities."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["embed", "similarity"]},
                    "text": {"type": "string"},
                    "paths": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="comm_bioacoustic",
            description=(
                "Analyze animal/nature audio with Earth Species Project's "
                "NatureLM-audio when installed; otherwise returns enable "
                "instructions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["analyze"]},
                    "audio_path": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["action"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "comm_identify_modality":
            return _identify_modality(arguments)
        if name == "comm_infotheory":
            return _infotheory(arguments)
        if name == "comm_discover":
            return await _discover(arguments)
        if name == "comm_language":
            return _language(arguments)
        if name == "comm_signal":
            return _signal(arguments)
        if name == "comm_quantum":
            return _quantum(arguments)
        if name == "comm_embed":
            return _embed(arguments)
        if name == "comm_bioacoustic":
            return _bioacoustic(arguments)
        return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error in {name}: {type(e).__name__}: {e}")]


def _out(payload) -> list[TextContent]:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str)
    return [TextContent(type="text", text=truncate(text))]


# ---- core tools -----------------------------------------------------------
def _identify_modality(args: dict) -> list[TextContent]:
    from src.comms_codecs import signal_profile
    signal = args.get("signal", "")
    if not signal:
        return _out("Error: 'signal' is required.")
    profile = signal_profile(signal)
    guesses = []
    if profile["looks_binary"]:
        guesses.append("binary (8-bit chunks → bytes)")
    if profile["looks_hex"]:
        guesses.append("hexadecimal bytes")
    if profile["looks_morse"]:
        guesses.append("morse code")
    ent = profile["entropy_bits_per_symbol"]
    if not guesses:
        if ent < 1.0:
            guesses.append("highly redundant / near-constant (control signal or padding)")
        elif ent <= 4.8:
            guesses.append("natural-language-like text or simple cipher")
        elif ent < 6.5:
            guesses.append("encoded text (base64/compressed) or structured data")
        else:
            guesses.append("random-looking: encrypted, compressed, or noise")
    profile["modality_hypotheses"] = guesses
    profile["note"] = (
        "Run comm_discover to attempt decoding, or comm_embed/comm_signal/"
        "comm_quantum for richer modality handling."
    )
    return _out(profile)


def _infotheory(args: dict) -> list[TextContent]:
    import src.comms_infotheory as it
    action = args.get("action", "")
    if action == "entropy":
        data = _symbols(args.get("data", ""))
        return _out({
            "action": "entropy",
            "entropy_bits_per_symbol": round(it.entropy_of_sequence(data), 4),
            "alphabet_size": len(set(data)),
            "length": len(data),
            "redundancy": round(it.estimate_redundancy(data), 4),
        })
    sent = _symbols(args.get("sent", ""))
    received = _symbols(args.get("received", ""))
    if not sent or not received:
        return _out("Error: 'sent' and 'received' are required for this action.")
    if action == "mutual_information":
        return _out({"action": action,
                     "mutual_information_bits": round(it.mutual_information(sent, received), 4),
                     "normalized_mutual_information": round(
                         it.normalized_mutual_information(sent, received), 4)})
    if action == "channel_capacity":
        return _out({"action": action,
                     "channel_capacity_bits_per_use": round(it.channel_capacity(sent, received), 4)})
    if action == "channel_established":
        threshold = float(args.get("threshold", it.DEFAULT_NMI_THRESHOLD))
        return _out(it.channel_established(sent, received, threshold=threshold))
    return _out(f"Error: unknown action '{action}'.")


async def _discover(args: dict) -> list[TextContent]:
    signal = args.get("signal", "")
    if not signal:
        return _out("Error: 'signal' is required.")
    url, model, headers = _resolve_llm()
    if not url or not model:
        return _out(
            "Error: no LLM endpoint is configured, which the discovery loop needs "
            "to propose and invent methods. Configure a chat model in Settings, "
            "then retry. (You can still run comm_identify_modality and "
            "comm_infotheory without a model.)"
        )
    from src.protocol_discovery import ProtocolDiscoverer
    disc = ProtocolDiscoverer(
        llm_endpoint=url, llm_model=model, llm_headers=headers,
        max_rounds=int(args.get("max_rounds", 3)),
        max_time=int(args.get("max_time", 45)),
        mi_threshold=float(args.get("threshold", 0.5)),
    )
    result = await disc.discover(signal, context=args.get("context", ""))
    # Compact the attempts log for tool output.
    result["attempts"] = [
        {k: a[k] for k in ("round", "mode", "method",
                           "normalized_mutual_information", "established") if k in a}
        for a in result.get("attempts", [])
    ]
    return _out(result)


# ---- modality tools (lazy / graceful) -------------------------------------
def _language(args: dict) -> list[TextContent]:
    action = args.get("action", "")
    # Transcription can reuse the repo's existing optional faster-whisper path.
    if action == "transcribe":
        mod, err = _try_import("faster_whisper", "Speech transcription",
                               "faster-whisper")
        if err:
            return _out(err)
        return _out("faster-whisper is available; wire transcription via the "
                    "existing STT service (services/stt). Provide 'audio_path'.")
    mod, err = _try_import("transformers", "Cross-language translation/detection",
                           "transformers sentencepiece "
                           "# then load facebook/seamless-m4t-v2-large or NLLB-200")
    if err:
        return _out(err)
    return _out(
        f"transformers is available. For '{action}', load SeamlessM4T "
        "(facebook/seamless-m4t-v2-large) for speech+text across ~100 languages, "
        "or NLLB-200 (facebook/nllb-200-distilled-600M) for text across 200. "
        "Milestone 2 wires the model call here."
    )


def _signal(args: dict) -> list[TextContent]:
    action = args.get("action", "")
    scheme = args.get("scheme", "BPSK")
    mod, err = _try_import("sk_dsp_comm", "Optical/RF physical-layer processing",
                           "scikit-dsp-comm")
    analytic = {
        "analyze": "Estimate symbol rate, SNR, and constellation; inspect the "
                   "power spectrum for the modulation footprint.",
        "demodulate": f"For {scheme}: carrier/phase sync, matched filter, then "
                      "symbol decision → bits.",
        "decode_fec": "Apply Viterbi (convolutional) or belief-propagation (LDPC) "
                      "decoding to recover the payload from coded bits.",
    }.get(action, "Unknown action.")
    if err:
        return _out(f"{err}\n\nAnalytic guidance (no library needed): {analytic}")
    return _out(f"scikit-dsp-comm available — {analytic} "
                "(Milestone 2 wires the digitalcomm/fec_conv calls here.)")


def _quantum(args: dict) -> list[TextContent]:
    action = args.get("action", "")
    protocol = (args.get("protocol", "") or "bb84").lower()
    descriptions = {
        "bb84": "BB84 QKD: Alice sends qubits in random bases (Z/X); Bob measures "
                "in random bases; they keep bits where bases matched (sifting) and "
                "estimate eavesdropping from the error rate. Security rests on the "
                "no-cloning theorem.",
        "teleportation": "Quantum teleportation: a shared Bell pair plus 2 classical "
                         "bits transfer an unknown qubit state; the state moves, the "
                         "particle does not.",
        "superdense": "Superdense coding: a shared Bell pair lets one qubit carry 2 "
                      "classical bits.",
    }
    if action == "describe_protocol":
        return _out({"protocol": protocol,
                     "description": descriptions.get(protocol,
                        "Unknown protocol. Known: bb84, teleportation, superdense.")})
    # simulate_channel
    mod, err = _try_import("squanch", "Quantum channel simulation",
                           "squanch  # or qunetsim / netsquid")
    if err:
        return _out(f"{err}\n\nProtocol summary: "
                    f"{descriptions.get(protocol, 'unknown protocol')}")
    return _out(f"squanch available — build the {protocol} circuit and a noisy "
                "QChannel to estimate QBER/fidelity. (Milestone 3 wires this.)")


def _embed(args: dict) -> list[TextContent]:
    mod, err = _try_import("imagebind", "Cross-modal (any-to-any) embedding",
                           "git+https://github.com/facebookresearch/ImageBind")
    if err:
        return _out(err)
    return _out("ImageBind available — embed text/image/audio into one space and "
                "compute cross-modal cosine similarity. (Milestone 2 wires this.)")


def _bioacoustic(args: dict) -> list[TextContent]:
    mod, err = _try_import("transformers", "Bioacoustic analysis (NatureLM-audio)",
                           "transformers  # then load EarthSpeciesProject/NatureLM-audio")
    if err:
        return _out(err)
    return _out("transformers available — load EarthSpeciesProject/NatureLM-audio "
                "and prompt it with the animal-audio question in 'query'. "
                "(Milestone 3 wires this.)")


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())
