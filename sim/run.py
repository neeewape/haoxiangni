from __future__ import annotations

import json
from pathlib import Path

from .data import align_futures_and_iv, load_futures_daily, load_option_iv_daily
from .monte_carlo import SimConfig, run_monthly_backtest, summarize_results
from .strategy import StrategyParams, add_proxy_sr_and_direction


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fut_path = root / "红枣期货数据.CSV"
    iv_path = root / "红枣期权数据.CSV"

    fut = load_futures_daily(fut_path)
    iv = load_option_iv_daily(iv_path)
    df = align_futures_and_iv(fut, iv)

    p = StrategyParams(
        sr_window_days=20,
        support_q=0.15,
        resistance_q=0.85,
        dir_window_days=60,
        tenor_days=21,
        hedge_every_days=1,
        exposure_frac=0.10,
        iv_mode="mean",
    )
    df = add_proxy_sr_and_direction(df, p)

    cfg = SimConfig(n_paths=5000, seed=42, capital_base=1.0)
    month_results = run_monthly_backtest(df, p=p, cfg=cfg, price_process="gbm_const_iv")
    summary = summarize_results(month_results)

    print("=== Summary (synthetic options, baseline) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary:
        print("\nStop-check (user thresholds):")
        median_ok = summary["median_monthly_return"] >= 0.01
        p_loss_ok = summary["p_loss"] <= 0.25
        print(f"- median>=1%: {median_ok} (median={summary['median_monthly_return']:.4%})")
        print(f"- p_loss<=25%: {p_loss_ok} (p_loss={summary['p_loss']:.2%})")


if __name__ == "__main__":
    main()

