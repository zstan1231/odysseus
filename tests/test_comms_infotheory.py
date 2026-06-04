"""Pure-numpy information-theory core for the Universal Communication Agent.

These tests pin the objective channel metrics that the discovery loop's stop
condition relies on. No heavy deps, deterministic.
"""
import random

from src.comms_infotheory import (
    channel_capacity,
    channel_established,
    entropy_of_sequence,
    estimate_redundancy,
    mutual_information,
    normalized_mutual_information,
    shannon_entropy,
)


def test_entropy_uniform_vs_deterministic():
    # Uniform over 4 symbols -> 2 bits; constant -> 0 bits.
    assert abs(shannon_entropy([0.25, 0.25, 0.25, 0.25]) - 2.0) < 1e-9
    assert shannon_entropy([1.0, 0.0, 0.0]) == 0.0
    assert entropy_of_sequence("aaaa") == 0.0
    assert abs(entropy_of_sequence("abcd" * 25) - 2.0) < 1e-9


def test_mutual_information_independent_vs_correlated():
    random.seed(0)
    sent = [random.randint(0, 3) for _ in range(2000)]
    perfect = list(sent)
    independent = [random.randint(0, 3) for _ in range(2000)]

    # Perfect copy: MI ~ H(X) ~ 2 bits, NMI ~ 1.
    assert mutual_information(sent, perfect) > 1.9
    assert normalized_mutual_information(sent, perfect) > 0.99
    # Independent: NMI near 0 (small finite-sample bias allowed).
    assert normalized_mutual_information(sent, independent) < 0.05


def test_channel_capacity_noiseless_binary():
    # Noiseless binary channel has capacity 1 bit/use.
    sent = [i % 2 for i in range(1000)]
    received = list(sent)
    assert abs(channel_capacity(sent, received) - 1.0) < 0.05


def test_channel_established_threshold_logic():
    random.seed(1)
    sent = [random.randint(0, 1) for _ in range(1000)]
    perfect = list(sent)
    # Flip ~40% of bits -> recovered information well below half.
    noisy = [b if random.random() > 0.4 else 1 - b for b in sent]
    independent = [random.randint(0, 1) for _ in range(1000)]

    assert channel_established(sent, perfect, threshold=0.5)["established"] is True
    assert channel_established(sent, independent, threshold=0.5)["established"] is False
    # A very low threshold should accept even a noisy-but-correlated channel.
    assert channel_established(sent, noisy, threshold=0.01)["established"] is True
    # The metrics dict always reports the numbers used for the decision.
    m = channel_established(sent, perfect)
    assert m["normalized_mutual_information"] >= m["threshold"]
    assert m["samples"] == len(sent)


def test_channel_established_empty_and_constant():
    assert channel_established([], [])["established"] is False
    # Sender constant, receiver constant -> nothing to recover, treated as ok.
    assert normalized_mutual_information("aaaa", "bbbb") == 1.0
    # Sender constant, receiver varies -> no recoverable mapping.
    assert normalized_mutual_information("aaaa", "abcd") == 0.0


def test_redundancy_constant_vs_uniform():
    assert estimate_redundancy("aaaaaa") == 1.0
    assert estimate_redundancy("abcd" * 50) < 0.05
