# src/comms_codecs.py
"""
A library of real, reversible decoders plus an information-theoretic
"communicativeness" scorer for the Universal Communication Agent.

This is the practical engine behind the most common real task: *somebody handed
you an unknown signal — recover the message.* The protocol-discovery loop
(src/protocol_discovery.py) proposes which method to try; this module actually
applies it and scores the result. Scoring is grounded in information theory
(printability, redundancy in the natural-language band, dictionary hits) so the
loop has an objective 0..1 signal rather than an LLM guess.

All decoders are pure-Python (stdlib only) so they work on the default install.
Heavy modality decoders (optical/RF/quantum/ML) live in the MCP server behind
lazy imports; this module is the always-available baseline.
"""
from __future__ import annotations

import base64
import binascii
import re
from typing import Callable, Dict, List, Optional

from src.comms_infotheory import entropy_of_sequence, estimate_redundancy

# A compact set of common English words — enough to detect that a decoded blob
# is natural language without shipping a dictionary. Intentionally small; the
# scorer degrades gracefully (printability still contributes) for other tongues.
_COMMON_WORDS = set(
    """the be to of and a in that have i it for not on with he as you do at this
    but his by from they we say her she or an will my one all would there their
    what so up out if about who get which go me when make can like time no just
    him know take people into year your good some could them see other than then
    now look only come its over think also back after use two how our work first
    well way even new want because any these give day most us is are was were has
    hello world message signal communication data code test water earth life""".split()
)

MORSE_TABLE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9",
}

_PRINTABLE = set(range(32, 127))


# ---------------------------------------------------------------------------
# Decoders. Each takes the raw signal string and returns decoded text, raising
# on structural failure (caller catches and records the method as inapplicable).
# ---------------------------------------------------------------------------
def _decode_identity(signal: str) -> str:
    return signal


def _decode_hex(signal: str) -> str:
    cleaned = re.sub(r"[^0-9a-fA-F]", "", signal)
    if len(cleaned) < 2:
        raise ValueError("not enough hex digits")
    if len(cleaned) % 2:
        cleaned = cleaned[:-1]
    return binascii.unhexlify(cleaned).decode("utf-8", errors="replace")


def _decode_base64(signal: str) -> str:
    cleaned = re.sub(r"\s+", "", signal)
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", cleaned or ""):
        raise ValueError("not base64 alphabet")
    pad = (-len(cleaned)) % 4
    return base64.b64decode(cleaned + "=" * pad).decode("utf-8", errors="replace")


def _decode_binary(signal: str) -> str:
    bits = re.sub(r"[^01]", "", signal)
    if len(bits) < 8:
        raise ValueError("not enough bits")
    bits = bits[: len(bits) - (len(bits) % 8)]
    chars = [chr(int(bits[i : i + 8], 2)) for i in range(0, len(bits), 8)]
    return "".join(chars)


def _decode_morse(signal: str) -> str:
    # Words separated by ' / ' or 3+ spaces; letters by single space.
    norm = signal.strip().replace("|", "/")
    words = re.split(r"\s*/\s*|\s{3,}", norm)
    out = []
    matched = False
    for w in words:
        letters = []
        for tok in w.split():
            if tok in MORSE_TABLE:
                letters.append(MORSE_TABLE[tok])
                matched = True
            elif tok:
                letters.append("?")
        out.append("".join(letters))
    if not matched:
        raise ValueError("no morse tokens recognized")
    return " ".join(p for p in out if p)


def _make_caesar(shift: int) -> Callable[[str], str]:
    def _decode(signal: str) -> str:
        res = []
        for ch in signal:
            if "a" <= ch <= "z":
                res.append(chr((ord(ch) - 97 - shift) % 26 + 97))
            elif "A" <= ch <= "Z":
                res.append(chr((ord(ch) - 65 - shift) % 26 + 65))
            else:
                res.append(ch)
        return "".join(res)

    return _decode


def _decode_atbash(signal: str) -> str:
    res = []
    for ch in signal:
        if "a" <= ch <= "z":
            res.append(chr(219 - ord(ch)))  # 219 = ord('a')+ord('z')
        elif "A" <= ch <= "Z":
            res.append(chr(155 - ord(ch)))  # 155 = ord('A')+ord('Z')
        else:
            res.append(ch)
    return "".join(res)


def _decode_reverse(signal: str) -> str:
    return signal[::-1]


# Registry: method-name -> decoder. ROT variants are generated on demand.
BASE_DECODERS: Dict[str, Callable[[str], str]] = {
    "identity": _decode_identity,
    "ascii": _decode_identity,
    "utf-8": _decode_identity,
    "hex": _decode_hex,
    "base64": _decode_base64,
    "binary": _decode_binary,
    "morse": _decode_morse,
    "atbash": _decode_atbash,
    "reverse": _decode_reverse,
    "rot13": _make_caesar(13),
}


def resolve_decoder(method_name: str) -> Optional[Callable[[str], str]]:
    """Map an LLM-proposed method name to a decoder, tolerating phrasing.

    Recognizes 'caesar shift 3', 'rot7', 'base-64', 'binary (8-bit)', etc.
    Returns None if no known decoder matches (the engine then treats it as a
    novel method to be invented/handled elsewhere).
    """
    name = (method_name or "").strip().lower()
    if name in BASE_DECODERS:
        return BASE_DECODERS[name]
    m = re.search(r"(?:rot|caesar|shift)\D*(\d{1,2})", name)
    if m:
        return _make_caesar(int(m.group(1)) % 26)
    for key in BASE_DECODERS:
        if key in name:
            return BASE_DECODERS[key]
    if "base" in name and "64" in name:
        return _decode_base64
    if "bin" in name:
        return _decode_binary
    return None


# ---------------------------------------------------------------------------
# Communicativeness: an information-theoretic 0..1 score for decoded text.
# ---------------------------------------------------------------------------
def communicativeness(text: str) -> Dict[str, float]:
    """Score how much a decoded string looks like an intentional message.

    Combines three signals, each grounded in information theory / language
    statistics: printable fraction, recognized-word fraction, and whether the
    per-character entropy sits in the natural-language band (~1–4.5 bits/char,
    well below the ~8 bits/char of random bytes). Returns a detail dict whose
    ``score`` is the headline 0..1 normalized-MI proxy used as the stop signal.
    """
    if not text:
        return {"score": 0.0, "printable": 0.0, "word_ratio": 0.0, "entropy": 0.0}

    printable = sum(1 for ch in text if ord(ch) in _PRINTABLE) / len(text)

    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if tokens:
        hit_chars = sum(len(t) for t in tokens if t in _COMMON_WORDS)
        total_alpha = sum(len(t) for t in tokens)
        word_ratio = hit_chars / total_alpha if total_alpha else 0.0
    else:
        word_ratio = 0.0

    h = entropy_of_sequence(list(text))
    # Language band bonus: reward entropy typical of natural text.
    if 1.0 <= h <= 4.8:
        band = 1.0
    elif h < 1.0:
        band = max(0.0, h)  # near-constant text is suspicious
    else:
        band = max(0.0, 1.0 - (h - 4.8) / 3.2)  # decays toward random-byte entropy

    score = 0.35 * printable + 0.5 * word_ratio + 0.15 * band
    return {
        "score": round(float(max(0.0, min(1.0, score))), 4),
        "printable": round(printable, 4),
        "word_ratio": round(word_ratio, 4),
        "entropy": round(h, 4),
    }


def builtin_attempt(method: Dict, signal: str) -> Dict:
    """Apply one proposed decoding method to the signal and score it.

    Returns the engine's attempt-result contract: a ``score`` (0..1 normalized-MI
    proxy via communicativeness), plus the decoded text and a human note. Methods
    the codec library doesn't recognize come back ``ok=False`` so the discovery
    loop knows to escalate into invent-a-protocol mode.
    """
    name = method.get("method") or method.get("name") or str(method)

    # Invented substitution ciphers: the discovery loop (in invent mode) can
    # propose an explicit char->char mapping, which we CAN execute and measure —
    # making invention partially functional even in the MVP, scored by the same
    # information-theoretic communicativeness metric as known codecs.
    mapping = method.get("mapping")
    if isinstance(mapping, dict) and mapping:
        try:
            table = {str(k)[:1]: str(v)[:1] for k, v in mapping.items() if str(k)}
            decoded = "".join(table.get(ch, ch) for ch in signal)
        except Exception as e:
            return {"method": name, "ok": False, "score": 0.0, "decoded": "",
                    "note": f"invented mapping failed: {e}"}
        detail = communicativeness(decoded)
        preview = decoded if len(decoded) <= 280 else decoded[:280] + "…"
        return {"method": name, "ok": True, "score": detail["score"],
                "decoded": preview, "invented": True,
                "note": f"invented substitution: word_ratio={detail['word_ratio']:.2f}",
                "detail": detail}

    decoder = resolve_decoder(name)
    if decoder is None:
        return {
            "method": name,
            "ok": False,
            "score": 0.0,
            "decoded": "",
            "note": f"no built-in decoder for '{name}' — candidate for invention",
        }
    try:
        decoded = decoder(signal)
    except Exception as e:  # structural failure = method doesn't apply
        return {
            "method": name,
            "ok": False,
            "score": 0.0,
            "decoded": "",
            "note": f"method '{name}' did not apply cleanly: {e}",
        }
    detail = communicativeness(decoded)
    preview = decoded if len(decoded) <= 280 else decoded[:280] + "…"
    return {
        "method": name,
        "ok": True,
        "score": detail["score"],
        "decoded": preview,
        "note": (
            f"printable={detail['printable']:.2f} "
            f"word_ratio={detail['word_ratio']:.2f} "
            f"entropy={detail['entropy']:.2f} bits/char"
        ),
        "detail": detail,
    }


def signal_profile(signal: str) -> Dict[str, object]:
    """Quick information-theoretic fingerprint of a raw signal for planning."""
    chars = list(signal or "")
    alphabet = sorted(set(chars))
    only = "".join(alphabet)
    looks_binary = bool(re.fullmatch(r"[01\s]+", signal or "")) and len(alphabet) <= 4
    looks_hex = bool(re.fullmatch(r"[0-9a-fA-F\s]+", signal or "")) and len(signal or "") >= 4
    looks_morse = bool(re.fullmatch(r"[.\-/|\s]+", signal or "")) and "." in (signal or "")
    return {
        "length": len(chars),
        "alphabet_size": len(alphabet),
        "alphabet_sample": only[:40],
        "entropy_bits_per_symbol": round(entropy_of_sequence(chars), 4),
        "redundancy": round(estimate_redundancy(chars), 4),
        "looks_binary": looks_binary,
        "looks_hex": looks_hex,
        "looks_morse": looks_morse,
    }
