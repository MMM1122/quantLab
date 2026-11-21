"""Deterministic content hashing for datasets and configs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode())


def dataframe_sha256(df: pd.DataFrame) -> str:
    """Hash of a DataFrame's *content*, independent of the Parquet encoder version.

    Columns are taken in sorted order, rows in their given order (callers canonicalise
    ordering first). Floats are serialised with repr-exact precision via ``float.hex``.
    """
    h = hashlib.sha256()
    cols = sorted(df.columns)
    h.update(("|".join(cols) + "\n").encode())
    for col in cols:
        s = df[col]
        h.update(f"#{col}:{s.dtype}\n".encode())
        if pd.api.types.is_float_dtype(s):
            vals = s.to_numpy(dtype="float64")
            h.update(np.ascontiguousarray(vals).tobytes())
        elif pd.api.types.is_datetime64_any_dtype(s):
            h.update(np.ascontiguousarray(s.to_numpy(dtype="datetime64[ns]").astype("int64")).tobytes())
        elif pd.api.types.is_integer_dtype(s):
            h.update(np.ascontiguousarray(s.to_numpy(dtype="int64")).tobytes())
        else:
            h.update("\x1f".join("" if pd.isna(v) else str(v) for v in s.tolist()).encode())
    return h.hexdigest()
