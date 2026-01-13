# Iteration 0 Summary

Generated: 2026-01-13T06:02:17

## Experiment Design
- Price process: gbm
- Calibration: iv_mean
- Tests: random_walk, direction_hit_rate
- Metrics: {"target_median_return": 0.01, "max_loss_prob": 0.25, "min_q05": -0.1}
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
  "iv_mode": "mean"
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
  "price_process": "gbm",
  "jump_intensity": 0.15,
  "jump_mean": -0.02,
  "jump_vol": 0.1
}
```

## Results Summary
```json
{
  "mean_monthly_return": 510.7856200106484,
  "median_monthly_return": 460.9058688099041,
  "p_loss": 0.00014736842105263158,
  "p_return_below_1pct": 0.00014736842105263158,
  "q05": 271.25447742268375,
  "q01": 215.91245644543974,
  "n_months_simulated": 81.0,
  "n_samples": 95000.0
}
```

## Diagnosis
Baseline metrics: median_return=46090.5869%, p_loss=0.01%, q05=27125.45%
