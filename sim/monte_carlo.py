from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bs import bs_price_and_delta
from .strategy import StrategyParams, pick_iv


@dataclass(frozen=True)
class SimConfig:
    n_paths: int = 10000
    seed: int = 42
    fee_per_fut_turnover: float = 0.0  # per 1 unit futures traded (placeholder)
    slippage_per_fut_turnover: float = 0.0
    capital_base: float = 1.0  # return normalization; 1 => return per unit notional


@dataclass(frozen=True)
class MonthResult:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    start_price: float
    iv_used: float
    strike_put: float
    strike_call: float
    direction: int
    pnl_paths: np.ndarray  # shape (n_paths,)


def _gbm_paths(s0: float, mu: float, sigma: float, n_steps: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    dt = 1.0 / 252.0
    z = rng.standard_normal((n_paths, n_steps))
    increments = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z
    log_s = np.log(s0) + np.cumsum(increments, axis=1)
    s = np.concatenate([np.full((n_paths, 1), s0), np.exp(log_s)], axis=1)
    return s  # (n_paths, n_steps+1)


def simulate_one_month(
    *,
    s0: float,
    k_put: float,
    k_call: float,
    iv: float,
    direction: int,
    p: StrategyParams,
    cfg: SimConfig,
) -> np.ndarray:
    """
    Synthetic short strangle + futures delta hedge, keeping exposure_frac directional delta.
    Uses Black-76 on futures with constant IV for the month (baseline).
    """
    rng = np.random.default_rng(cfg.seed)
    n_steps = p.tenor_days
    t_total = n_steps / 252.0
    dt = 1.0 / 252.0

    # For short-horizon, keep mu=0 (random walk). Direction is expressed via keeping net delta.
    mu = 0.0
    s_paths = _gbm_paths(s0, mu, iv, n_steps, cfg.n_paths, rng)

    # Initial option prices and deltas (t=t_total)
    call0, d_call0 = bs_price_and_delta(f=s0, k=k_call, t=t_total, sigma=iv, r=p.r, option_type="C")
    put0, d_put0 = bs_price_and_delta(f=s0, k=k_put, t=t_total, sigma=iv, r=p.r, option_type="P")

    # Short both options => receive premium
    premium = call0 + put0

    # Hedge with futures: choose futures position to hit target net delta
    # Net delta of short options = -(d_call0 + d_put0)
    target_delta = p.exposure_frac * float(direction)  # in [-0.1, 0, +0.1]
    fut_pos = target_delta - (-(d_call0 + d_put0))  # fut delta is 1 per unit
    fut_cost = 0.0  # futures PnL accumulates via price changes; assume zero carry
    turnover = 0.0

    # Step through time
    fut_pos_paths = np.full(cfg.n_paths, fut_pos, dtype=float)
    for step in range(1, n_steps + 1):
        s_t = s_paths[:, step]
        t_remain = max((n_steps - step) * dt, 0.0)

        if (step % p.hedge_every_days) == 0 and t_remain > 0:
            # Recompute deltas under same IV (synthetic)
            call_t, d_call_t = bs_price_and_delta(f=s_t, k=k_call, t=t_remain, sigma=iv, r=p.r, option_type="C")
            put_t, d_put_t = bs_price_and_delta(f=s_t, k=k_put, t=t_remain, sigma=iv, r=p.r, option_type="P")
            opt_net_delta = -(d_call_t + d_put_t)
            desired_fut = target_delta - opt_net_delta
            delta_trade = desired_fut - fut_pos_paths
            turnover += np.abs(delta_trade).mean()
            fut_pos_paths = desired_fut

    # Option payoff at expiry
    s_T = s_paths[:, -1]
    payoff_call = np.maximum(s_T - k_call, 0.0)
    payoff_put = np.maximum(k_put - s_T, 0.0)
    opt_pnl = premium - (payoff_call + payoff_put)  # short options

    # Futures PnL: integral of position * dS (discrete sum)
    # Approx with end position * (S_T - S0) is rough; do discrete sum with piecewise-constant positions.
    # For simplicity in this minimal version, approximate with average position over month:
    # (This will be refined by LLM iterations.)
    avg_pos = fut_pos_paths  # after last hedge; use as proxy
    fut_pnl = avg_pos * (s_T - s0)

    trading_cost = (cfg.fee_per_fut_turnover + cfg.slippage_per_fut_turnover) * turnover
    total_pnl = opt_pnl + fut_pnl - trading_cost

    # Normalize to "return per capital_base"
    return total_pnl / cfg.capital_base


def run_monthly_backtest(
    df: pd.DataFrame,
    *,
    p: StrategyParams,
    cfg: SimConfig,
) -> list[MonthResult]:
    """
    Walk through the futures history and run a synthetic 1-month simulation at each month start.
    """
    df = df.dropna(subset=["support", "resistance", "direction"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Not enough history after building proxy support/resistance and direction.")

    df["month"] = df["date"].dt.to_period("M")
    month_starts = df.groupby("month").head(1).reset_index(drop=True)

    results: list[MonthResult] = []
    for _, row in month_starts.iterrows():
        start_idx = int(row.name)
        # Ensure we have enough forward days
        if start_idx + p.tenor_days >= len(df):
            break

        s0 = float(row["close"])
        k_put = float(row["support"])
        k_call = float(row["resistance"])
        direction = int(row["direction"])
        iv = pick_iv(row, p.iv_mode)

        pnl_paths = simulate_one_month(
            s0=s0,
            k_put=k_put,
            k_call=k_call,
            iv=iv,
            direction=direction,
            p=p,
            cfg=cfg,
        )

        end_date = df.loc[start_idx + p.tenor_days, "date"]
        results.append(
            MonthResult(
                start_date=row["date"],
                end_date=end_date,
                start_price=s0,
                iv_used=iv,
                strike_put=k_put,
                strike_call=k_call,
                direction=direction,
                pnl_paths=pnl_paths,
            )
        )
    return results


def summarize_results(month_results: list[MonthResult]) -> dict[str, float]:
    all_rets = np.concatenate([mr.pnl_paths for mr in month_results]) if month_results else np.array([])
    all_rets = all_rets[~np.isnan(all_rets)]
    if all_rets.size == 0:
        return {}

    # Treat each path-month as a sample of monthly return
    median = float(np.median(all_rets))
    mean = float(np.mean(all_rets))
    p_loss = float(np.mean(all_rets < 0))
    p_below_1pct = float(np.mean(all_rets < 0.01))
    q05 = float(np.quantile(all_rets, 0.05))
    q01 = float(np.quantile(all_rets, 0.01))

    return {
        "mean_monthly_return": mean,
        "median_monthly_return": median,
        "p_loss": p_loss,
        "p_return_below_1pct": p_below_1pct,
        "q05": q05,
        "q01": q01,
        "n_months_simulated": float(len(month_results)),
        "n_samples": float(all_rets.size),
    }

