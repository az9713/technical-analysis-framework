"""Typed wrapper functions around pandas_ta for all Tier 1 indicators.

Every function accepts an OHLCV DataFrame and returns a new DataFrame with
the indicator column(s) appended.  NaN-leading rows are left as-is (the
caller or engine can choose to drop them).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name(result: pd.Series | None, name: str) -> pd.Series | None:
    """Safely set the name on a Series, returning None unchanged."""
    if result is not None:
        result.name = name
    return result


def _append(df: pd.DataFrame, result: pd.DataFrame | pd.Series | None, fallback_name: str | None = None) -> pd.DataFrame:
    """Append indicator result columns to *df* and return the combined frame.

    If *result* is ``None`` (pandas_ta returns None when data is too short),
    a column of NaN is added using *fallback_name*.
    """
    out = df.copy()
    if result is None:
        if fallback_name:
            out[fallback_name] = np.nan
        return out
    if isinstance(result, pd.Series):
        out[result.name] = result
    elif isinstance(result, pd.DataFrame):
        for col in result.columns:
            out[col] = result[col]
    return out


# ===================================================================
# TREND
# ===================================================================

def sma(df: pd.DataFrame, length: int = 20, column: str = "close") -> pd.DataFrame:
    """Simple Moving Average."""
    col_name = f"SMA_{length}"
    result = _name(ta.sma(df[column], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def ema(df: pd.DataFrame, length: int = 20, column: str = "close") -> pd.DataFrame:
    """Exponential Moving Average."""
    col_name = f"EMA_{length}"
    result = _name(ta.ema(df[column], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def wma(df: pd.DataFrame, length: int = 20, column: str = "close") -> pd.DataFrame:
    """Weighted Moving Average."""
    col_name = f"WMA_{length}"
    result = _name(ta.wma(df[column], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def dema(df: pd.DataFrame, length: int = 20, column: str = "close") -> pd.DataFrame:
    """Double Exponential Moving Average."""
    col_name = f"DEMA_{length}"
    result = _name(ta.dema(df[column], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def tema(df: pd.DataFrame, length: int = 20, column: str = "close") -> pd.DataFrame:
    """Triple Exponential Moving Average."""
    col_name = f"TEMA_{length}"
    result = _name(ta.tema(df[column], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def vwma(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Volume Weighted Moving Average."""
    col_name = f"VWMA_{length}"
    result = _name(ta.vwma(df["close"], df["volume"], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def hma(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Hull Moving Average."""
    col_name = f"HMA_{length}"
    result = _name(ta.hma(df["close"], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def kama(df: pd.DataFrame, length: int = 10, fast: int = 2, slow: int = 30) -> pd.DataFrame:
    """Kaufman Adaptive Moving Average."""
    col_name = f"KAMA_{length}_{fast}_{slow}"
    result = _name(ta.kama(df["close"], length=length, fast=fast, slow=slow), col_name)
    return _append(df, result, fallback_name=col_name)


def t3(df: pd.DataFrame, length: int = 5, a: float = 0.7) -> pd.DataFrame:
    """Tillson T3 Moving Average."""
    col_name = f"T3_{length}"
    result = _name(ta.t3(df["close"], length=length, a=a), col_name)
    return _append(df, result, fallback_name=col_name)


def supertrend(df: pd.DataFrame, length: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend indicator."""
    result = ta.supertrend(df["high"], df["low"], df["close"], length=length, multiplier=multiplier)
    return _append(df, result, fallback_name=f"SUPERT_{length}_{multiplier}")


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    """Ichimoku Cloud."""
    raw = ta.ichimoku(df["high"], df["low"], df["close"], tenkan=tenkan, kijun=kijun, senkou=senkou)
    if raw is None or (isinstance(raw, tuple) and raw[0] is None):
        return df.copy()
    ichi = raw[0] if isinstance(raw, tuple) else raw
    return _append(df, ichi)


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Average Directional Index."""
    result = ta.adx(df["high"], df["low"], df["close"], length=length)
    return _append(df, result, fallback_name=f"ADX_{length}")


def aroon(df: pd.DataFrame, length: int = 25) -> pd.DataFrame:
    """Aroon indicator."""
    result = ta.aroon(df["high"], df["low"], length=length)
    return _append(df, result, fallback_name=f"AROON_{length}")


def psar(df: pd.DataFrame, af0: float = 0.02, af: float = 0.02, max_af: float = 0.2) -> pd.DataFrame:
    """Parabolic SAR."""
    result = ta.psar(df["high"], df["low"], df["close"], af0=af0, af=af, max_af=max_af)
    return _append(df, result, fallback_name="PSAR")


# ===================================================================
# MOMENTUM
# ===================================================================

def rsi(df: pd.DataFrame, length: int = 14, column: str = "close") -> pd.DataFrame:
    """Relative Strength Index."""
    col_name = f"RSI_{length}"
    result = _name(ta.rsi(df[column], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence."""
    result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    return _append(df, result, fallback_name=f"MACD_{fast}_{slow}_{signal}")


def stoch(df: pd.DataFrame, k: int = 14, d: int = 3, smooth_k: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator."""
    result = ta.stoch(df["high"], df["low"], df["close"], k=k, d=d, smooth_k=smooth_k)
    return _append(df, result, fallback_name=f"STOCHk_{k}_{d}_{smooth_k}")


def cci(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Commodity Channel Index."""
    col_name = f"CCI_{length}"
    result = _name(ta.cci(df["high"], df["low"], df["close"], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def willr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Williams %R."""
    col_name = f"WILLR_{length}"
    result = _name(ta.willr(df["high"], df["low"], df["close"], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def roc(df: pd.DataFrame, length: int = 10) -> pd.DataFrame:
    """Rate of Change."""
    col_name = f"ROC_{length}"
    result = _name(ta.roc(df["close"], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def mfi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Money Flow Index."""
    col_name = f"MFI_{length}"
    result = _name(ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def tsi(df: pd.DataFrame, fast: int = 13, slow: int = 25) -> pd.DataFrame:
    """True Strength Index."""
    result = ta.tsi(df["close"], fast=fast, slow=slow)
    return _append(df, result, fallback_name=f"TSI_{fast}_{slow}")


def uo(df: pd.DataFrame, fast: int = 7, medium: int = 14, slow: int = 28) -> pd.DataFrame:
    """Ultimate Oscillator."""
    col_name = f"UO_{fast}_{medium}_{slow}"
    result = _name(ta.uo(df["high"], df["low"], df["close"], fast=fast, medium=medium, slow=slow), col_name)
    return _append(df, result, fallback_name=col_name)


def ao(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.DataFrame:
    """Awesome Oscillator."""
    col_name = f"AO_{fast}_{slow}"
    result = _name(ta.ao(df["high"], df["low"], fast=fast, slow=slow), col_name)
    return _append(df, result, fallback_name=col_name)


# ===================================================================
# VOLATILITY
# ===================================================================

def bbands(df: pd.DataFrame, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands."""
    result = ta.bbands(df["close"], length=length, std=std)
    return _append(df, result, fallback_name=f"BBL_{length}_{std}")


def atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Average True Range."""
    col_name = f"ATR_{length}"
    result = _name(ta.atr(df["high"], df["low"], df["close"], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def kc(df: pd.DataFrame, length: int = 20, scalar: float = 1.5) -> pd.DataFrame:
    """Keltner Channel."""
    result = ta.kc(df["high"], df["low"], df["close"], length=length, scalar=scalar)
    return _append(df, result, fallback_name=f"KCL_{length}_{scalar}")


def donchian(df: pd.DataFrame, lower_length: int = 20, upper_length: int = 20) -> pd.DataFrame:
    """Donchian Channel."""
    result = ta.donchian(df["high"], df["low"], lower_length=lower_length, upper_length=upper_length)
    return _append(df, result, fallback_name=f"DCL_{lower_length}_{upper_length}")


def stdev(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Standard Deviation of close prices."""
    col_name = f"STDEV_{length}"
    result = _name(ta.stdev(df["close"], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def true_range(df: pd.DataFrame) -> pd.DataFrame:
    """True Range."""
    result = _name(ta.true_range(df["high"], df["low"], df["close"]), "TRUERANGE")
    return _append(df, result, fallback_name="TRUERANGE")


# ===================================================================
# VOLUME
# ===================================================================

def obv(df: pd.DataFrame) -> pd.DataFrame:
    """On Balance Volume."""
    result = _name(ta.obv(df["close"], df["volume"]), "OBV")
    return _append(df, result, fallback_name="OBV")


def vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Weighted Average Price."""
    result = _name(ta.vwap(df["high"], df["low"], df["close"], df["volume"]), "VWAP")
    return _append(df, result, fallback_name="VWAP")


def ad(df: pd.DataFrame) -> pd.DataFrame:
    """Accumulation/Distribution Line."""
    result = _name(ta.ad(df["high"], df["low"], df["close"], df["volume"]), "AD")
    return _append(df, result, fallback_name="AD")


def cmf(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Chaikin Money Flow."""
    col_name = f"CMF_{length}"
    result = _name(ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def fi(df: pd.DataFrame, length: int = 13) -> pd.DataFrame:
    """Force Index."""
    col_name = f"FI_{length}"
    result = _name(ta.efi(df["close"], df["volume"], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


def volume_sma(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Volume Simple Moving Average."""
    col_name = f"VOLUME_SMA_{length}"
    result = _name(ta.sma(df["volume"], length=length), col_name)
    return _append(df, result, fallback_name=col_name)


# ===================================================================
# OVERLAP
# ===================================================================

def pivot(df: pd.DataFrame, method: str = "traditional") -> pd.DataFrame:
    """Pivot Points (uses previous bar's HLC)."""
    # Compute classic pivot points from prior period
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_close = df["close"].shift(1)

    pp = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)

    out = df.copy()
    out["PIVOT_PP"] = pp
    out["PIVOT_R1"] = r1
    out["PIVOT_S1"] = s1
    out["PIVOT_R2"] = r2
    out["PIVOT_S2"] = s2
    return out


def fib(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Fibonacci Retracement Levels based on rolling high/low."""
    roll_high = df["high"].rolling(window=length).max()
    roll_low = df["low"].rolling(window=length).min()
    diff = roll_high - roll_low

    out = df.copy()
    out["FIB_0.0"] = roll_low
    out["FIB_23.6"] = roll_low + 0.236 * diff
    out["FIB_38.2"] = roll_low + 0.382 * diff
    out["FIB_50.0"] = roll_low + 0.500 * diff
    out["FIB_61.8"] = roll_low + 0.618 * diff
    out["FIB_100.0"] = roll_high
    return out


# ---------------------------------------------------------------------------
# Registry of wrapper functions keyed by indicator name
# ---------------------------------------------------------------------------

WRAPPER_MAP: dict[str, callable] = {
    "sma": sma, "ema": ema, "wma": wma, "dema": dema, "tema": tema,
    "vwma": vwma, "hma": hma, "kama": kama, "t3": t3,
    "supertrend": supertrend, "ichimoku": ichimoku, "adx": adx,
    "aroon": aroon, "psar": psar,
    "rsi": rsi, "macd": macd, "stoch": stoch, "cci": cci,
    "willr": willr, "roc": roc, "mfi": mfi, "tsi": tsi, "uo": uo, "ao": ao,
    "bbands": bbands, "atr": atr, "kc": kc, "donchian": donchian,
    "stdev": stdev, "true_range": true_range,
    "obv": obv, "vwap": vwap, "ad": ad, "cmf": cmf, "fi": fi,
    "volume_sma": volume_sma,
    "pivot": pivot, "fib": fib,
}
