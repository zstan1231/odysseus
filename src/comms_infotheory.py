# src/comms_infotheory.py
"""
Information-theory core for the Universal Communication Agent.

This is the *objective substrate* that decides whether a communication channel
has actually been established between two parties — independent of any LLM
judgment. It is deliberately pure-Python + numpy (numpy is already a core
Odysseus dependency) so it always works on the default install, with zero heavy
optical/quantum/ML libraries present.

Everything here measures one question in different ways: **how much of what was
sent actually made it across?** That is mutual information. When the discovery
loop (src/protocol_discovery.py) invents a new encoding, this module scores it.

Symbols may be any hashable values (ints, str characters, tuples). Sequences are
aligned pairs: ``sent[i]`` was transmitted and ``received[i]`` was observed.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

# Default threshold (in *normalized* mutual information, 0..1) above which we
# consider a channel "established" — i.e. the receiver recovers at least half of
# the information in the sent signal. Configurable per call.
DEFAULT_NMI_THRESHOLD = 0.5
_EPS = 1e-12


def _as_list(seq: Sequence[Hashable]) -> List[Hashable]:
    return list(seq)


def shannon_entropy(probs: Sequence[float], base: float = 2.0) -> float:
    """Entropy H(p) = -Σ p·log(p) of a probability distribution, in bits (base 2).

    Tolerates unnormalized inputs (normalizes first) and zero entries.
    """
    p = np.asarray([x for x in probs if x is not None], dtype=float)
    if p.size == 0:
        return 0.0
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    nz = p[p > _EPS]
    return float(-np.sum(nz * (np.log(nz) / np.log(base))))


def distribution(symbols: Sequence[Hashable]) -> Dict[Hashable, float]:
    """Empirical probability distribution of a symbol sequence."""
    symbols = _as_list(symbols)
    n = len(symbols)
    if n == 0:
        return {}
    counts = Counter(symbols)
    return {s: c / n for s, c in counts.items()}


def entropy_of_sequence(symbols: Sequence[Hashable], base: float = 2.0) -> float:
    """Shannon entropy of a raw symbol sequence (bits/symbol)."""
    dist = distribution(symbols)
    return shannon_entropy(list(dist.values()), base=base)


def _joint_counts(
    sent: Sequence[Hashable], received: Sequence[Hashable]
) -> Tuple[np.ndarray, List[Hashable], List[Hashable]]:
    """Build a joint count matrix C[i, j] = #(sent==x_i & received==y_j)."""
    sent = _as_list(sent)
    received = _as_list(received)
    if len(sent) != len(received):
        raise ValueError(
            f"sent and received must be the same length "
            f"({len(sent)} != {len(received)})"
        )
    x_alphabet = sorted(set(sent), key=lambda v: (str(type(v)), str(v)))
    y_alphabet = sorted(set(received), key=lambda v: (str(type(v)), str(v)))
    x_index = {s: i for i, s in enumerate(x_alphabet)}
    y_index = {s: j for j, s in enumerate(y_alphabet)}
    mat = np.zeros((len(x_alphabet), len(y_alphabet)), dtype=float)
    for a, b in zip(sent, received):
        mat[x_index[a], y_index[b]] += 1.0
    return mat, x_alphabet, y_alphabet


def mutual_information(
    sent: Sequence[Hashable], received: Sequence[Hashable], base: float = 2.0
) -> float:
    """Mutual information I(X;Y) between aligned sent/received sequences (bits).

    I(X;Y) = Σ p(x,y) · log( p(x,y) / (p(x)·p(y)) ). Zero when received is
    independent of sent; equal to H(X) when received perfectly determines sent.
    """
    counts, _, _ = _joint_counts(sent, received)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p_xy = counts / total
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    denom = p_x @ p_y  # outer product p(x)p(y)
    mask = p_xy > _EPS
    ratio = np.zeros_like(p_xy)
    ratio[mask] = np.log(p_xy[mask] / denom[mask]) / np.log(base)
    return float(np.sum(p_xy[mask] * ratio[mask]))


def normalized_mutual_information(
    sent: Sequence[Hashable], received: Sequence[Hashable]
) -> float:
    """I(X;Y) normalized to 0..1 by the sent-signal entropy H(X).

    This is the natural "fraction of the message recovered" score. 1.0 means the
    receiver can perfectly reconstruct what was sent; 0.0 means no information
    crossed. When H(X)=0 (sender sent a constant) there is nothing to recover, so
    we return 1.0 by convention if received is also constant, else 0.0.
    """
    sent = _as_list(sent)
    received = _as_list(received)
    h_x = entropy_of_sequence(sent)
    if h_x <= _EPS:
        # Degenerate sender. "Established" only if the receiver is also stable.
        return 1.0 if entropy_of_sequence(received) <= _EPS else 0.0
    mi = mutual_information(sent, received)
    return float(max(0.0, min(1.0, mi / h_x)))


def channel_matrix(
    sent: Sequence[Hashable], received: Sequence[Hashable]
) -> Tuple[np.ndarray, List[Hashable], List[Hashable]]:
    """Empirical channel transition matrix P(received | sent).

    Row i is the conditional distribution over outputs given input symbol x_i.
    """
    counts, x_alpha, y_alpha = _joint_counts(sent, received)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid div-by-zero for unseen inputs
    return counts / row_sums, x_alpha, y_alpha


def blahut_arimoto(
    p_y_given_x: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-7,
    base: float = 2.0,
) -> float:
    """Capacity (bits/use) of a discrete memoryless channel via Blahut-Arimoto.

    Capacity = max over input distributions of I(X;Y). This is the *theoretical
    best* any code over this channel could achieve, which is why the discovery
    loop reports it alongside the measured MI: it tells you how much headroom a
    better encoding would have. Pure numpy, deterministic, no external solver.
    """
    q = np.asarray(p_y_given_x, dtype=float)
    if q.ndim != 2 or q.size == 0:
        return 0.0
    # Normalize rows to be valid conditional distributions.
    row_sums = q.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    q = q / row_sums
    n_in = q.shape[0]
    r = np.full(n_in, 1.0 / n_in)  # input distribution, start uniform

    prev_capacity = -1.0
    for _ in range(max_iter):
        # q_ji = P(x_i | y_j) via Bayes given current input dist r
        py = r @ q  # output distribution
        py = np.where(py <= _EPS, _EPS, py)
        # Q[i, j] = r_i q_ij / py_j  (posterior of input given output)
        posterior = (r[:, None] * q) / py[None, :]
        # Update r_i ∝ exp(Σ_j q_ij log Q_ij)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(q > _EPS, np.log(np.where(posterior > _EPS, posterior, _EPS)), 0.0)
        exponent = np.sum(q * log_term, axis=1)
        r_new = np.exp(exponent)
        r_new /= r_new.sum()
        # Capacity estimate at this step.
        py_new = r_new @ q
        py_new = np.where(py_new <= _EPS, _EPS, py_new)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.sum(
                r_new[:, None]
                * q
                * np.where(q > _EPS, np.log(q / py_new[None, :]), 0.0)
            )
        capacity = float(mi / np.log(base))
        r = r_new
        if abs(capacity - prev_capacity) < tol:
            break
        prev_capacity = capacity
    return max(0.0, capacity)


def channel_capacity(
    sent: Sequence[Hashable], received: Sequence[Hashable], base: float = 2.0
) -> float:
    """Estimate channel capacity (bits/use) from observed sent/received pairs."""
    mat, _, _ = channel_matrix(sent, received)
    return blahut_arimoto(mat, base=base)


def channel_established(
    sent: Sequence[Hashable],
    received: Sequence[Hashable],
    threshold: float = DEFAULT_NMI_THRESHOLD,
) -> Dict[str, object]:
    """The decision the discovery loop hinges on: is a channel established?

    Returns a metrics dict so callers can both branch on ``established`` and
    display the numbers. ``established`` is True when the receiver recovers at
    least ``threshold`` fraction of the sent signal's information (normalized
    mutual information) — an objective, non-hallucinated stop signal.
    """
    sent = _as_list(sent)
    received = _as_list(received)
    n = min(len(sent), len(received))
    if n == 0:
        return {
            "established": False,
            "reason": "no aligned symbols to measure",
            "samples": 0,
            "mutual_information_bits": 0.0,
            "normalized_mutual_information": 0.0,
            "sent_entropy_bits": 0.0,
            "received_entropy_bits": 0.0,
            "channel_capacity_bits": 0.0,
            "threshold": threshold,
        }
    sent, received = sent[:n], received[:n]
    h_x = entropy_of_sequence(sent)
    h_y = entropy_of_sequence(received)
    mi = mutual_information(sent, received)
    nmi = normalized_mutual_information(sent, received)
    capacity = channel_capacity(sent, received)
    established = nmi >= threshold
    if established:
        reason = (
            f"receiver recovers {nmi:.0%} of the sent signal "
            f"(≥ {threshold:.0%} threshold)"
        )
    else:
        reason = (
            f"receiver recovers only {nmi:.0%} of the sent signal "
            f"(< {threshold:.0%} threshold)"
        )
    return {
        "established": bool(established),
        "reason": reason,
        "samples": n,
        "mutual_information_bits": round(mi, 4),
        "normalized_mutual_information": round(nmi, 4),
        "sent_entropy_bits": round(h_x, 4),
        "received_entropy_bits": round(h_y, 4),
        "channel_capacity_bits": round(capacity, 4),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Symbolization helpers — turn arbitrary payloads into comparable symbol
# sequences so the measures above can be applied to text, bytes, or numbers.
# ---------------------------------------------------------------------------
def symbolize(payload: object) -> List[Hashable]:
    """Coerce an arbitrary payload into a list of discrete symbols.

    - str  -> list of characters
    - bytes/bytearray -> list of int byte values
    - list/tuple -> elements as-is (already symbols)
    - anything else -> characters of its str() form
    """
    if isinstance(payload, (list, tuple)):
        return list(payload)
    if isinstance(payload, (bytes, bytearray)):
        return list(payload)
    if isinstance(payload, str):
        return list(payload)
    return list(str(payload))


def estimate_redundancy(symbols: Sequence[Hashable]) -> float:
    """Redundancy = 1 - H(X)/log2(|alphabet|), in 0..1.

    High redundancy in an *unknown* signal is a strong hint that it is a
    structured/intentional message rather than noise — used by the modality
    identifier and the discovery planner.
    """
    symbols = _as_list(symbols)
    dist = distribution(symbols)
    k = len(dist)
    if k <= 1:
        return 1.0
    h = shannon_entropy(list(dist.values()))
    h_max = math.log2(k)
    if h_max <= _EPS:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - h / h_max)))
