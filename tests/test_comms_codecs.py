"""Codec library + communicativeness scorer for the Universal Communication Agent."""
import base64
import codecs

from src.comms_codecs import (
    builtin_attempt,
    communicativeness,
    resolve_decoder,
    signal_profile,
)

_MSG = "hello world this is a message about the water and life on earth"


def test_builtin_attempt_decodes_base64():
    sig = base64.b64encode(_MSG.encode()).decode()
    res = builtin_attempt({"method": "base64"}, sig)
    assert res["ok"] is True
    assert res["score"] >= 0.5
    assert "hello world" in res["decoded"]


def test_builtin_attempt_decodes_binary_and_rot13():
    bits = "".join(format(b, "08b") for b in _MSG.encode())
    assert builtin_attempt({"method": "binary (8-bit)"}, bits)["score"] >= 0.5
    rot = codecs.encode(_MSG, "rot13")
    assert builtin_attempt({"method": "rot13"}, rot)["score"] >= 0.5


def test_correct_method_beats_identity_on_encoded_signal():
    # Decoding an encoded signal with the right method must look more
    # communicative than leaving it opaque under identity.
    sig = base64.b64encode(_MSG.encode()).decode()
    identity = builtin_attempt({"method": "identity"}, sig)["score"]
    decoded = builtin_attempt({"method": "base64"}, sig)["score"]
    assert decoded > identity


def test_unknown_method_marked_not_ok():
    res = builtin_attempt({"method": "quantum entanglement handshake"}, "abc")
    assert res["ok"] is False
    assert res["score"] == 0.0


def test_invented_mapping_is_executable():
    # 'e' replaced by 'X' in the plaintext; inventing X->e should recover it.
    garbled = _MSG.replace("e", "X")
    res = builtin_attempt({"method": "invented", "mapping": {"X": "e"}}, garbled)
    assert res["ok"] is True
    assert res.get("invented") is True
    assert res["score"] >= 0.5


def test_resolve_decoder_tolerates_phrasing():
    assert resolve_decoder("caesar shift 3") is not None
    assert resolve_decoder("base-64") is not None
    assert resolve_decoder("ROT7") is not None
    assert resolve_decoder("totally unknown scheme") is None


def test_communicativeness_language_vs_noise():
    lang = communicativeness("the quick brown fox jumps over the lazy dog")
    noise = communicativeness("x9#q2@!zkv%^&*<>")
    assert lang["score"] > noise["score"]


def test_signal_profile_flags():
    assert signal_profile("0101 0101 1100")["looks_binary"] is True
    assert signal_profile("48 65 6c 6c 6f")["looks_hex"] is True
    assert signal_profile(".... . .-.. .-.. ---")["looks_morse"] is True
