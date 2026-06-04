"""Robustness of the discovery loop's JSON proposal parser.

Weak local models echo the prompt's Example array, wrap output in code fences,
or append prose. The parser must still recover the real method objects — the
same resilience deep_research relies on (see test_deep_research_parse_*).
"""
from src.protocol_discovery import ProtocolDiscoverer


def _p(text):
    return ProtocolDiscoverer("http://x", "m")._parse_json_objects(text)


def test_parses_plain_array():
    objs = _p('[{"method": "base64", "rationale": "looks b64"}]')
    assert objs == [{"method": "base64", "rationale": "looks b64"}]


def test_strips_code_fence():
    objs = _p('```json\n[{"method": "hex"}]\n```')
    assert objs[0]["method"] == "hex"


def test_keeps_real_array_after_echoed_example():
    # Model echoes the Example: [...] then emits its real answer.
    text = ('Sure! Example: [{"method": "base64", "rationale": "x"}]\n'
            'Here are my picks:\n[{"method": "morse"}, {"method": "binary"}]')
    objs = _p(text)
    methods = [o.get("method") for o in objs]
    assert methods == ["morse", "binary"]


def test_recovers_standalone_objects_when_no_array():
    text = 'I suggest {"method": "rot13"} and also {"method": "atbash"}.'
    methods = sorted(o.get("method") for o in _p(text))
    assert methods == ["atbash", "rot13"]


def test_returns_empty_on_garbage():
    assert _p("no json here at all") == []


def test_preserves_mapping_field_for_inventions():
    objs = _p('[{"method": "invented", "mapping": {"X": "e"}}]')
    assert objs[0]["mapping"] == {"X": "e"}
