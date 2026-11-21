"""User-facing labels that must accompany results. Kept in one place so tests can assert on them."""

SYNTHETIC_LABEL = "SYNTHETIC — NOT REAL MARKET PERFORMANCE"
BACKTEST_LABEL = "BACKTEST / SIMULATION — NOT REAL PERFORMANCE"
REPLAY_LABEL = "HISTORICAL REPLAY — NOT LIVE MARKET CONDITIONS"
RESEARCH_ONLY_LABEL = "RESEARCH-ONLY COUNTERFACTUAL — NOT A TRADABLE POLICY"
NOT_ADVICE = "For research and education only. Not investment advice."


def data_label(is_synthetic: bool) -> str:
    return SYNTHETIC_LABEL if is_synthetic else "REAL MARKET DATA (provider snapshot)"
