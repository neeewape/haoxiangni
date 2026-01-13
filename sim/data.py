from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FuturesBar:
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


def _read_csv_guess_encoding(path: Path) -> pd.DataFrame:
    """
    The provided CSVs look like GBK/GB2312 exported with Chinese headers.
    We'll try a small set of common encodings.
    """
    encodings = ["utf-8-sig", "gbk", "gb2312", "cp936"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001 - intentional
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path} with encodings {encodings}: {last_err}")


def _to_float_series(s: pd.Series) -> pd.Series:
    # Handles values like "1,310,602" or "8,820.00"
    return (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', "", regex=False)
        .replace("nan", pd.NA)
        .astype(float)
    )


def load_futures_daily(csv_path: str | Path) -> pd.DataFrame:
    """
    Expect columns (in Chinese) like:
    - 日期
    - 开盘价(元/吨) 最高价(元/吨) 最低价(元/吨) 收盘价(元/吨) 成交量(手)
    """
    p = Path(csv_path)
    df = _read_csv_guess_encoding(p)

    # Normalize column names by stripping whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # First column is date
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    # Try to map expected price/volume columns by substring
    def col_contains(substr: str) -> str:
        for c in df.columns:
            if substr in c:
                return c
        raise KeyError(f"Could not find column containing '{substr}' in {df.columns.tolist()}")

    open_col = col_contains("开盘")
    high_col = col_contains("最高")
    low_col = col_contains("最低")
    close_col = col_contains("收盘")
    vol_col = col_contains("成交量")

    out = pd.DataFrame(
        {
            "date": df[date_col],
            "open": _to_float_series(df[open_col]),
            "high": _to_float_series(df[high_col]),
            "low": _to_float_series(df[low_col]),
            "close": _to_float_series(df[close_col]),
            "volume": _to_float_series(df[vol_col]),
        }
    )
    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return out


def load_option_iv_daily(csv_path: str | Path) -> pd.DataFrame:
    """
    Expect columns (Chinese headers provided by user):
    - 交易日期
    - 日均隐含波动率(%)
    - 日最低隐含波动率(%)
    - 日最高隐含波动率(%)
    - 日波动率标准差(%)
    - 日合约数量
    - 成交量(手)
    """
    p = Path(csv_path)
    df = _read_csv_guess_encoding(p)
    df.columns = [str(c).strip() for c in df.columns]

    date_col = "交易日期" if "交易日期" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    def must(name: str) -> str:
        if name not in df.columns:
            raise KeyError(f"Missing column '{name}' in {df.columns.tolist()}")
        return name

    out = pd.DataFrame(
        {
            "date": df[date_col],
            "iv_mean": _to_float_series(df[must("日均隐含波动率(%)")]) / 100.0,
            "iv_min": _to_float_series(df[must("日最低隐含波动率(%)")]) / 100.0,
            "iv_max": _to_float_series(df[must("日最高隐含波动率(%)")]) / 100.0,
            "iv_std": _to_float_series(df[must("日波动率标准差(%)")]) / 100.0,
            "n_contracts": _to_float_series(df[must("日合约数量")]),
            "opt_volume": _to_float_series(df[must("成交量(手)")]),
        }
    )
    out = out.dropna(subset=["date", "iv_mean"]).sort_values("date").reset_index(drop=True)
    return out


def align_futures_and_iv(fut: pd.DataFrame, iv: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join IV onto futures by date. If IV is missing for a date, forward-fill
    to keep simulation continuous (still reported as an assumption).
    """
    df = fut.merge(iv, on="date", how="left")
    df[["iv_mean", "iv_min", "iv_max", "iv_std"]] = df[
        ["iv_mean", "iv_min", "iv_max", "iv_std"]
    ].ffill()
    return df

