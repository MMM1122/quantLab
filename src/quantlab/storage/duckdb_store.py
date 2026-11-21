"""DuckDB analytics over immutable Parquet snapshots (read-only queries)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def _con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def query_bars(
    parquet_path: str | Path,
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    sql = "SELECT * FROM read_parquet(?) WHERE 1=1"
    params: list = [str(parquet_path)]
    if symbols:
        sql += " AND symbol IN (" + ",".join("?" for _ in symbols) + ")"
        params += list(symbols)
    if start:
        sql += " AND session_date >= CAST(? AS TIMESTAMP)"
        params.append(start)
    if end:
        sql += " AND session_date <= CAST(? AS TIMESTAMP)"
        params.append(end)
    sql += " ORDER BY symbol, session_date"
    with _con() as con:
        return con.execute(sql, params).df()


def symbol_summary(parquet_path: str | Path) -> pd.DataFrame:
    sql = """
        SELECT symbol,
               COUNT(*)                          AS n_bars,
               MIN(session_date)                 AS first_session,
               MAX(session_date)                 AS last_session,
               SUM(CASE WHEN dividend > 0 THEN 1 ELSE 0 END) AS n_dividends,
               SUM(dividend)                     AS total_dividends,
               SUM(CASE WHEN split_ratio <> 1 THEN 1 ELSE 0 END) AS n_splits,
               AVG(volume)                       AS avg_volume
        FROM read_parquet(?)
        GROUP BY symbol ORDER BY symbol
    """
    with _con() as con:
        return con.execute(sql, [str(parquet_path)]).df()
