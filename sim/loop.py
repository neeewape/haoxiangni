from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import math

from .data import align_futures_and_iv, load_futures_daily, load_option_iv_daily
from .monte_carlo import SimConfig, run_monthly_backtest, summarize_results
from .strategy import StrategyParams, add_proxy_sr_and_direction

STOP_CRITERIA = {
    "target_median_return": 0.01,
    "max_loss_prob": 0.25,
    "min_q05": -0.10,
    "min_iters": 3,
    "convergence_rounds": 3,
    "convergence_epsilon": 0.001,
    "max_iters": 8,
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(path: Path, *, iteration: int, params: StrategyParams, cfg: SimConfig, summary: dict, diagnosis: str) -> None:
    lines = [
        f"# Iteration {iteration} Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Experiment Design (baseline)",
        "- Price process: short-horizon GBM with constant IV",
        "- Strategy: short strangle near support/resistance + delta hedge",
        "- Exposure: keep 10% directional delta aligned with proxy signal",
        "- Costs: futures fee/slippage from config",
        "",
        "## Strategy Params",
        "```json",
        json.dumps(asdict(params), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Simulation Config",
        "```json",
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Results Summary",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Diagnosis",
        diagnosis,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_stop(summary: dict) -> tuple[bool, str]:
    if not summary:
        return False, "empty_summary"
    if any(math.isnan(summary[key]) for key in ("median_monthly_return", "p_loss", "q05")):
        return False, "nan_metrics"
    median_ok = summary["median_monthly_return"] >= STOP_CRITERIA["target_median_return"]
    p_loss_ok = summary["p_loss"] <= STOP_CRITERIA["max_loss_prob"]
    q05_ok = summary["q05"] >= STOP_CRITERIA["min_q05"]
    if median_ok and p_loss_ok and q05_ok:
        return True, "target_met"
    return False, "target_not_met"


def _adjust_params(params: StrategyParams, summary: dict) -> tuple[StrategyParams, str]:
    support_q = params.support_q
    resistance_q = params.resistance_q
    exposure_frac = params.exposure_frac
    hedge_every_days = params.hedge_every_days
    iv_mode = params.iv_mode

    changes: list[str] = []
    median_ret = summary.get("median_monthly_return", float("nan"))
    p_loss = summary.get("p_loss", float("nan"))
    q05 = summary.get("q05", float("nan"))

    if math.isnan(median_ret) or median_ret < STOP_CRITERIA["target_median_return"]:
        support_q = min(support_q + 0.02, 0.30)
        resistance_q = max(resistance_q - 0.02, 0.70)
        changes.append("tighten strikes to lift premium")
        iv_mode = "max" if iv_mode != "max" else iv_mode

    if math.isnan(p_loss) or math.isnan(q05) or p_loss > STOP_CRITERIA["max_loss_prob"] or q05 < STOP_CRITERIA["min_q05"]:
        support_q = max(support_q - 0.02, 0.10)
        resistance_q = min(resistance_q + 0.02, 0.90)
        exposure_frac = max(exposure_frac - 0.02, 0.06)
        hedge_every_days = max(1, hedge_every_days - 1)
        changes.append("widen strikes + reduce exposure to cut tail risk")

    if support_q >= resistance_q:
        support_q = 0.15
        resistance_q = 0.85
        changes.append("reset quantiles to preserve ordering")

    new_params = StrategyParams(
        sr_window_days=params.sr_window_days,
        support_q=support_q,
        resistance_q=resistance_q,
        dir_window_days=params.dir_window_days,
        tenor_days=params.tenor_days,
        hedge_every_days=hedge_every_days,
        exposure_frac=exposure_frac,
        r=params.r,
        iv_mode=iv_mode,
    )
    return new_params, "; ".join(changes) if changes else "no_change"


def _log_change(path: Path, iteration: int, reason: str, params: StrategyParams) -> None:
    entry = [
        f"## Iter {iteration}",
        f"- Change reason: {reason}",
        "- Strategy params:",
        "```json",
        json.dumps(asdict(params), ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry))


def run_loop() -> None:
    root = Path(__file__).resolve().parents[1]
    fut_path = root / "红枣期货数据.CSV"
    iv_path = root / "红枣期权数据.CSV"

    fut = load_futures_daily(fut_path)
    iv = load_option_iv_daily(iv_path)
    df = align_futures_and_iv(fut, iv)

    params = StrategyParams(
        sr_window_days=20,
        support_q=0.15,
        resistance_q=0.85,
        dir_window_days=60,
        tenor_days=21,
        hedge_every_days=1,
        exposure_frac=0.10,
        iv_mode="mean",
    )
    cfg = SimConfig(n_paths=5000, seed=42, capital_base=1.0)

    metrics_path = root / "define_metrics.md"
    change_log_path = root / "change_log.md"

    best_median = None
    stagnant_rounds = 0

    for iteration in range(STOP_CRITERIA["max_iters"]):
        df_iter = add_proxy_sr_and_direction(df, params)
        month_results = run_monthly_backtest(df_iter, p=params, cfg=cfg)
        summary = summarize_results(month_results)

        diagnosis_parts = [
            f"median_return={summary.get('median_monthly_return', float('nan')):.4%}",
            f"p_loss={summary.get('p_loss', float('nan')):.2%}",
            f"q05={summary.get('q05', float('nan')):.2%}",
        ]
        diagnosis = "Baseline metrics: " + ", ".join(diagnosis_parts)

        iter_dir = root / "runs" / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            iter_dir / "config.json",
            {
                "iteration": iteration,
                "strategy_params": asdict(params),
                "sim_config": asdict(cfg),
                "stop_criteria": STOP_CRITERIA,
            },
        )
        _write_summary(
            iter_dir / "summary.md",
            iteration=iteration,
            params=params,
            cfg=cfg,
            summary=summary,
            diagnosis=diagnosis,
        )

        stop, reason = _evaluate_stop(summary)
        if best_median is None:
            best_median = summary.get("median_monthly_return", 0.0)
        else:
            delta = summary.get("median_monthly_return", 0.0) - best_median
            if abs(delta) < STOP_CRITERIA["convergence_epsilon"]:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
                best_median = max(best_median, summary.get("median_monthly_return", 0.0))

        if stop and iteration + 1 >= STOP_CRITERIA["min_iters"]:
            _log_change(change_log_path, iteration, f"stop: {reason}", params)
            break
        if stagnant_rounds >= STOP_CRITERIA["convergence_rounds"]:
            _log_change(change_log_path, iteration, "stop: convergence", params)
            break

        params, reason = _adjust_params(params, summary)
        _log_change(change_log_path, iteration, reason, params)

    if metrics_path.exists():
        print(f"Metrics frozen at: {metrics_path}")
    print("Loop complete")


if __name__ == "__main__":
    run_loop()
