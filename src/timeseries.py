import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller


def rolling_view(minutes, cfg):
    window = cfg["timeseries"]["rolling_window"]
    trend_window = cfg["timeseries"]["trend_window"]
    view = pd.DataFrame(index=minutes.index)
    view["requests"] = minutes["requests"]
    view["requests_roll_mean"] = minutes["requests"].rolling(window, min_periods=1).mean()
    view["requests_roll_std"] = minutes["requests"].rolling(window, min_periods=2).std().fillna(0.0)
    view["requests_trend"] = minutes["requests"].rolling(trend_window, min_periods=10, center=True).median()
    view["latency_p95"] = minutes["latency_p95"]
    view["latency_roll_mean"] = minutes["latency_p95"].rolling(window, min_periods=1).mean()
    view["error_rate"] = minutes["error_rate"]
    view["error_roll_mean"] = minutes["error_rate"].rolling(window, min_periods=1).mean()
    return view


def resample_profiles(minutes):
    hourly = minutes.resample("1h").agg({
        "requests": "sum",
        "error_rate": "mean",
        "latency_p95": "mean",
    })
    by_hour = minutes.groupby(minutes.index.hour)["requests"].mean()
    by_dow = minutes.groupby(minutes.index.dayofweek)["requests"].mean()
    return hourly, by_hour, by_dow


def decompose(minutes, cfg):
    period = cfg["timeseries"]["seasonal_period"]
    series = minutes["requests"].astype(float)
    stl = STL(series, period=period, robust=True).fit()
    return pd.DataFrame({
        "observed": series,
        "trend": stl.trend,
        "seasonal": stl.seasonal,
        "resid": stl.resid,
    })


def linear_trend(series):
    x = np.arange(len(series), dtype=float)
    y = series.astype(float).values
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def summarize(minutes, parts, cfg):
    series = minutes["requests"].astype(float)
    slope, intercept = linear_trend(series)
    stat, pvalue = adfuller(series.values, autolag="AIC")[:2]
    correlations = acf(series.values, nlags=1500, fft=True)
    seasonal_strength = 1.0 - parts["resid"].var() / (parts["seasonal"] + parts["resid"]).var()
    trend_strength = 1.0 - parts["resid"].var() / (parts["trend"] + parts["resid"]).var()
    return {
        "points": int(len(series)),
        "frequency": cfg["timeseries"]["freq"],
        "mean_requests_per_minute": float(series.mean()),
        "std_requests_per_minute": float(series.std()),
        "linear_trend_per_minute": slope,
        "linear_trend_per_day": slope * 1440.0,
        "linear_intercept": intercept,
        "adf_statistic": float(stat),
        "adf_pvalue": float(pvalue),
        "acf_lag_1": float(correlations[1]),
        "acf_lag_60": float(correlations[60]),
        "acf_lag_1440": float(correlations[1440]),
        "seasonal_strength": float(seasonal_strength),
        "trend_strength": float(trend_strength),
        "peak_hour": int(minutes.groupby(minutes.index.hour)["requests"].mean().idxmax()),
        "trough_hour": int(minutes.groupby(minutes.index.hour)["requests"].mean().idxmin()),
    }
