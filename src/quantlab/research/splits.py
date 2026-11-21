"""Chronological splits for research (no shuffling, optional embargo between segments)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlab.core.errors import InputValidationError


@dataclass(frozen=True)
class Segment:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_sessions: int


def chronological_split(
    sessions: pd.DatetimeIndex, train: float = 0.6, validation: float = 0.2, embargo: int = 5
) -> list[Segment]:
    if not (0 < train < 1 and 0 <= validation < 1 and train + validation < 1):
        raise InputValidationError("fractions must satisfy 0<train, 0<=validation, train+validation<1")
    n = len(sessions)
    a = int(n * train)
    b = int(n * (train + validation))
    parts = [("train", 0, a), ("validation", a + embargo, b), ("test", b + embargo, n)]
    out = []
    for name, i, j in parts:
        if j - i <= 0:
            if name == "validation" and validation == 0:
                continue
            raise InputValidationError(f"segment {name} is empty; use more data or a smaller embargo")
        out.append(Segment(name, sessions[i], sessions[j - 1], j - i))
    return out


def walk_forward(
    sessions: pd.DatetimeIndex, train_size: int, test_size: int, step: int | None = None, embargo: int = 0,
    expanding: bool = True,
) -> list[tuple[Segment, Segment]]:
    step = step or test_size
    out = []
    i = train_size
    while i + embargo + test_size <= len(sessions):
        tr0 = 0 if expanding else i - train_size
        train = Segment("train", sessions[tr0], sessions[i - 1], i - tr0)
        t0 = i + embargo
        test = Segment("test", sessions[t0], sessions[t0 + test_size - 1], test_size)
        out.append((train, test))
        i += step
    if not out:
        raise InputValidationError("not enough sessions for one walk-forward fold")
    return out
