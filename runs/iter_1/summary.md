# Iteration 1 Summary

Generated: 2026-01-13T06:02:19

## Experiment Design
- Price process: jump
- Calibration: iv_max
- Tests: random_walk, tail_stress, direction_hit_rate
- Metrics: {"target_median_return": 0.012, "max_loss_prob": 0.22, "min_q05": -0.08}
- Strategy: short strangle near support/resistance + delta hedge
- Exposure: keep 10% directional delta aligned with proxy signal
- Costs: futures fee/slippage from config

## Strategy Params
```json
{
  "sr_window_days": 20,
  "support_q": 0.15,
  "resistance_q": 0.85,
  "dir_window_days": 60,
  "tenor_days": 21,
  "hedge_every_days": 1,
  "exposure_frac": 0.1,
  "r": 0.02,
  "iv_mode": "max"
}
```

## Simulation Config
```json
{
  "n_paths": 5000,
  "seed": 42,
  "fee_per_fut_turnover": 0.0,
  "slippage_per_fut_turnover": 0.0,
  "capital_base": 1.0,
  "price_process": "jump",
  "jump_intensity": 0.15,
  "jump_mean": -0.02,
  "jump_vol": 0.1
}
```

## Results Summary
```json
{
  "mean_monthly_return": 860.7855152576404,
  "median_monthly_return": 812.5599405144646,
  "p_loss": 0.00015789473684210527,
  "p_return_below_1pct": 0.00015789473684210527,
  "q05": 473.1400876801927,
  "q01": 372.2133623895324,
  "n_months_simulated": 81.0,
  "n_samples": 95000.0
}
```

## Diagnosis
Baseline metrics: median_return=81255.9941%, p_loss=0.02%, q05=47314.01%
