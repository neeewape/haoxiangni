from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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

PRICE_PROCESS_OPTIONS = (
    "gbm_const_iv",
    "gbm_stress_iv_up",
    "gbm_stress_iv_down",
    "gbm_drift_bias",
    "jump_diffusion_proxy",
)
CALIBRATION_OPTIONS = ("iv_mean", "iv_max", "iv_min", "iv_std")
VALIDATION_OPTIONS = ("monthly_backtest", "monthly_backtest_with_stress")


@dataclass
class ExperimentDesign:
    price_process: str
    calibration_method: str
    validation_method: str
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _default_design() -> ExperimentDesign:
    return ExperimentDesign(
        price_process="gbm_const_iv",
        calibration_method="iv_mean",
        validation_method="monthly_backtest",
        metrics={
            "target_median_return": STOP_CRITERIA["target_median_return"],
            "max_loss_prob": STOP_CRITERIA["max_loss_prob"],
            "min_q05": STOP_CRITERIA["min_q05"],
            "max_p_return_below_1pct": 0.55,
        },
        notes=["baseline_design"],
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(
    path: Path,
    *,
    iteration: int,
    params: StrategyParams,
    cfg: SimConfig,
    design: ExperimentDesign,
    summary: dict,
    diagnosis: str,
) -> None:
    lines = [
        f"# Iteration {iteration} Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Experiment Design",
        "```json",
        json.dumps(asdict(design), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Baseline Strategy",
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


def _evaluate_stop(summary: dict, metrics: dict[str, float]) -> tuple[bool, str]:
    if not summary:
        return False, "empty_summary"
    required = ("median_monthly_return", "p_loss", "q05")
    if any(math.isnan(summary[key]) for key in required):
        return False, "nan_metrics"
    median_ok = summary["median_monthly_return"] >= metrics.get("target_median_return", STOP_CRITERIA["target_median_return"])
    p_loss_ok = summary["p_loss"] <= metrics.get("max_loss_prob", STOP_CRITERIA["max_loss_prob"])
    q05_ok = summary["q05"] >= metrics.get("min_q05", STOP_CRITERIA["min_q05"])
    p_below_ok = summary["p_return_below_1pct"] <= metrics.get("max_p_return_below_1pct", 1.0)
    q01_ok = summary["q01"] >= metrics.get("min_q01", -1.0)
    if median_ok and p_loss_ok and q05_ok and p_below_ok and q01_ok:
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


def _adjust_design(design: ExperimentDesign, summary: dict, *, iteration: int) -> tuple[ExperimentDesign, str]:
    notes = list(design.notes)
    metrics = dict(design.metrics)
    price_process = PRICE_PROCESS_OPTIONS[(iteration + 1) % len(PRICE_PROCESS_OPTIONS)]
    calibration_method = CALIBRATION_OPTIONS[(iteration + 1) % len(CALIBRATION_OPTIONS)]
    validation_method = VALIDATION_OPTIONS[(iteration + 1) % len(VALIDATION_OPTIONS)]

    median_ret = summary.get("median_monthly_return", float("nan"))
    p_loss = summary.get("p_loss", float("nan"))
    q05 = summary.get("q05", float("nan"))

    metrics["target_median_return"] = min(metrics.get("target_median_return", 0.01) + 0.002, 0.02)
    metrics["max_loss_prob"] = max(metrics.get("max_loss_prob", 0.25) - 0.02, 0.10)
    metrics["min_q01"] = max(metrics.get("min_q01", -0.20) + 0.01, -0.10)
    metrics["max_p_return_below_1pct"] = max(metrics.get("max_p_return_below_1pct", 0.55) - 0.05, 0.30)

    notes.append("rotate_design_each_iter")
    notes.append(f"median_ret={median_ret:.4%}")
    notes.append(f"p_loss={p_loss:.2%}")
    notes.append(f"q05={q05:.2%}")

    new_design = ExperimentDesign(
        price_process=price_process,
        calibration_method=calibration_method,
        validation_method=validation_method,
        metrics=metrics,
        notes=notes,
    )
    return new_design, "design_update"


def _apply_design_to_params(design: ExperimentDesign, params: StrategyParams) -> StrategyParams:
    iv_mode = {
        "iv_mean": "mean",
        "iv_max": "max",
        "iv_min": "min",
        "iv_std": "std",
    }.get(design.calibration_method, params.iv_mode)
    if iv_mode == params.iv_mode:
        return params
    return StrategyParams(
        sr_window_days=params.sr_window_days,
        support_q=params.support_q,
        resistance_q=params.resistance_q,
        dir_window_days=params.dir_window_days,
        tenor_days=params.tenor_days,
        hedge_every_days=params.hedge_every_days,
        exposure_frac=params.exposure_frac,
        r=params.r,
        iv_mode=iv_mode,
    )


def _run_validation(
    df,
    *,
    params: StrategyParams,
    cfg: SimConfig,
    design: ExperimentDesign,
) -> dict:
    summary = summarize_results(
        run_monthly_backtest(df, p=params, cfg=cfg, price_process=design.price_process)
    )
    if design.validation_method != "monthly_backtest_with_stress":
        return summary

    stress_design = ExperimentDesign(
        price_process="gbm_stress_iv_up",
        calibration_method=design.calibration_method,
        validation_method=design.validation_method,
        metrics=design.metrics,
        notes=design.notes + ["stress_run"],
    )
    stress_summary = summarize_results(
        run_monthly_backtest(df, p=params, cfg=cfg, price_process=stress_design.price_process)
    )
    merged = dict(summary)
    merged.update({f"stress_{key}": value for key, value in stress_summary.items()})
    return merged


def _log_change(
    path: Path,
    iteration: int,
    reason: str,
    params: StrategyParams,
    design: ExperimentDesign,
) -> None:
    entry = [
        f"## Iter {iteration}",
        f"- Change reason: {reason}",
        "- Experiment design:",
        "```json",
        json.dumps(asdict(design), ensure_ascii=False, indent=2),
        "```",
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
    design = _default_design()

    metrics_path = root / "define_metrics.md"
    change_log_path = root / "change_log.md"

    best_median = None
    stagnant_rounds = 0

    for iteration in range(STOP_CRITERIA["max_iters"]):
        params = _apply_design_to_params(design, params)
        df_iter = add_proxy_sr_and_direction(df, params)
        summary = _run_validation(df_iter, params=params, cfg=cfg, design=design)

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
        _write_json(iter_dir / "experiment_design.json", asdict(design))
        _write_summary(
            iter_dir / "summary.md",
            iteration=iteration,
            params=params,
            cfg=cfg,
            design=design,
            summary=summary,
            diagnosis=diagnosis,
        )

        stop, reason = _evaluate_stop(summary, design.metrics)
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
            _log_change(change_log_path, iteration, f"stop: {reason}", params, design)
            break
        if stagnant_rounds >= STOP_CRITERIA["convergence_rounds"]:
            _log_change(change_log_path, iteration, "stop: convergence", params, design)
            break

        params, reason = _adjust_params(params, summary)
        design, design_reason = _adjust_design(design, summary, iteration=iteration)
        _log_change(change_log_path, iteration, f"{reason}; {design_reason}", params, design)

    if metrics_path.exists():
        print(f"Metrics frozen at: {metrics_path}")
    print("Loop complete")


if __name__ == "__main__":
    run_loop()
