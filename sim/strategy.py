from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyParams:
    # Support/resistance proxy: rolling window and quantiles
    sr_window_days: int = 20
    support_q: float = 0.15
    resistance_q: float = 0.85

    # Direction proxy: MA slope window
    dir_window_days: int = 60

    # Option tenor (approx trading days)
    tenor_days: int = 21

    # Hedging frequency in days
    hedge_every_days: int = 1

    # Keep 10% exposure aligned with direction proxy
    exposure_frac: float = 0.10

    # Risk-free for discounting in Black-76
    r: float = 0.02

    # Use which IV for baseline
    iv_mode: str = "mean"  # mean|min|max


def add_proxy_sr_and_direction(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """
    Given futures+iv dataframe with 'close', build:
    - support/resistance as rolling quantiles of close
    - direction as sign of MA slope (up/down/flat)
    """
    out = df.copy()
    close = out["close"].astype(float)

    out["support"] = close.rolling(p.sr_window_days, min_periods=p.sr_window_days).quantile(p.support_q)
    out["resistance"] = close.rolling(p.sr_window_days, min_periods=p.sr_window_days).quantile(p.resistance_q)

    ma = close.rolling(p.dir_window_days, min_periods=p.dir_window_days).mean()
    slope = ma.diff()
    # +1 up, -1 down, 0 flat (deadband by median abs slope)
    dead = slope.abs().rolling(p.dir_window_days, min_periods=p.dir_window_days).median()
    dead = dead.replace(0, np.nan).ffill()
    dir_raw = np.where(slope > dead, 1, np.where(slope < -dead, -1, 0))
    out["direction"] = pd.Series(dir_raw, index=out.index).astype(int)
    return out


def pick_iv(row: pd.Series, mode: str) -> float:
    mode = mode.lower()
    if mode == "mean":
        return float(row["iv_mean"])
    if mode == "min":
        return float(row["iv_min"])
    if mode == "max":
        return float(row["iv_max"])
    raise ValueError("iv_mode must be one of: mean|min|max")

