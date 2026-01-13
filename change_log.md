# 变更记录

## Iter 0
- 初始化迭代框架与指标口径。
## Iter 0
- Change reason: no_change
- Experiment design:
```json
{
  "price_process": "gbm",
  "calibration": "iv_mean",
  "tests": [
    "random_walk",
    "direction_hit_rate"
  ],
  "metrics": {
    "target_median_return": 0.01,
    "max_loss_prob": 0.25,
    "min_q05": -0.1
  }
}
```
- Strategy params:
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
## Iter 1
- Change reason: no_change
- Experiment design:
```json
{
  "price_process": "jump",
  "calibration": "iv_max",
  "tests": [
    "random_walk",
    "tail_stress",
    "direction_hit_rate"
  ],
  "metrics": {
    "target_median_return": 0.012,
    "max_loss_prob": 0.22,
    "min_q05": -0.08
  }
}
```
- Strategy params:
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
## Iter 2
- Change reason: stop: target_met
- Experiment design:
```json
{
  "price_process": "gbm",
  "calibration": "iv_min",
  "tests": [
    "random_walk",
    "vol_sensitivity"
  ],
  "metrics": {
    "target_median_return": 0.008,
    "max_loss_prob": 0.28,
    "min_q05": -0.12
  }
}
```
- Strategy params:
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
  "iv_mode": "min"
}
```
