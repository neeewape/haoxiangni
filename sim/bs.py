from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_price_and_delta(
    *,
    f: float | np.ndarray,
    k: float,
    t: float,
    sigma: float,
    r: float = 0.02,
    option_type: str,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Black (Black-76) on futures: option on futures with forward F.
    Vectorized over `f` (scalar or numpy array).

    Price = exp(-r t) * [ F N(d1) - K N(d2) ] for call; put via parity.
    Delta is with respect to futures price F.

    option_type: 'C' or 'P'
    """
    opt = option_type.upper()
    f_arr = np.asarray(f, dtype=float)

    if t <= 0:
        if opt == "C":
            intrinsic = np.maximum(f_arr - k, 0.0)
            delta = np.where(f_arr > k, 1.0, 0.0)
        elif opt == "P":
            intrinsic = np.maximum(k - f_arr, 0.0)
            delta = np.where(f_arr < k, -1.0, 0.0)
        else:
            raise ValueError("option_type must be 'C' or 'P'")
        # Preserve scalar type if input scalar
        if np.isscalar(f):
            return float(intrinsic), float(delta)
        return intrinsic, delta

    sigma = max(float(sigma), 1e-8)
    f_arr = np.maximum(f_arr, 1e-8)
    k = max(float(k), 1e-8)

    vol_sqrt = sigma * np.sqrt(t)
    d1 = (np.log(f_arr / k) + 0.5 * sigma * sigma * t) / vol_sqrt
    d2 = d1 - vol_sqrt
    disc = float(np.exp(-r * t))

    if opt == "C":
        price = disc * (f_arr * norm.cdf(d1) - k * norm.cdf(d2))
        delta = disc * norm.cdf(d1)
    elif opt == "P":
        price = disc * (k * norm.cdf(-d2) - f_arr * norm.cdf(-d1))
        delta = -disc * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'C' or 'P'")

    if np.isscalar(f):
        return float(price), float(delta)
    return price, delta

