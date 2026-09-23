import numpy as np
import pandas as pd

BASE_COLUMNS = [
    "requests",
    "error_rate",
    "server_error_rate",
    "latency_p50",
    "latency_p95",
    "latency_mean",
    "bytes_per_request",
    "unique_users",
]


def aggregate_events(events, freq="1min"):
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["is_error"] = (frame["status"] >= 400).astype(np.int8)
    frame["is_server_error"] = (frame["status"] >= 500).astype(np.int8)
    grouped = frame.set_index("timestamp").resample(freq)
    out = pd.DataFrame({
        "requests": grouped["status"].size(),
        "error_rate": grouped["is_error"].mean(),
        "server_error_rate": grouped["is_server_error"].mean(),
        "latency_p50": grouped["latency_ms"].quantile(0.50),
        "latency_p95": grouped["latency_ms"].quantile(0.95),
        "latency_mean": grouped["latency_ms"].mean(),
        "bytes_out": grouped["bytes_out"].sum(),
        "unique_users": grouped["user_id"].nunique(),
    })
    out["bytes_per_request"] = out["bytes_out"] / out["requests"].replace(0, np.nan)
    out = out.fillna(0.0)
    out.index.name = "minute"
    return out


def partial_aggregate(records):
    latency = np.array([r["latency_ms"] for r in records], dtype=float)
    status = np.array([r["status"] for r in records], dtype=int)
    payload = np.array([r["bytes_out"] for r in records], dtype=float)
    users = {r["user_id"] for r in records}
    return {
        "requests": int(latency.size),
        "errors": int((status >= 400).sum()),
        "server_errors": int((status >= 500).sum()),
        "latency_sum": float(latency.sum()),
        "latency_values": latency.tolist(),
        "bytes_out": float(payload.sum()),
        "users": list(users),
    }


def merge_partials(partials):
    requests = sum(p["requests"] for p in partials)
    if requests == 0:
        return {name: 0.0 for name in BASE_COLUMNS}
    latency = np.concatenate([np.asarray(p["latency_values"], dtype=float) for p in partials])
    users = set()
    for p in partials:
        users.update(p["users"])
    return {
        "requests": float(requests),
        "error_rate": sum(p["errors"] for p in partials) / requests,
        "server_error_rate": sum(p["server_errors"] for p in partials) / requests,
        "latency_p50": float(np.quantile(latency, 0.50)),
        "latency_p95": float(np.quantile(latency, 0.95)),
        "latency_mean": float(latency.mean()),
        "bytes_per_request": sum(p["bytes_out"] for p in partials) / requests,
        "unique_users": float(len(users)),
    }


def build_features(minutes, cfg):
    lags = cfg["ml"]["lags"]
    windows = cfg["ml"]["roll_windows"]
    frame = minutes[BASE_COLUMNS].copy()
    feat = pd.DataFrame(index=frame.index)

    for column in ["requests", "error_rate", "latency_p95", "latency_p50", "server_error_rate"]:
        for lag in lags:
            feat[f"{column}_lag{lag}"] = frame[column].shift(lag - 1) if lag > 1 else frame[column]
    for column in ["requests", "latency_p95", "error_rate"]:
        for window in windows:
            roll = frame[column].rolling(window, min_periods=2)
            feat[f"{column}_mean{window}"] = roll.mean()
            feat[f"{column}_std{window}"] = roll.std().fillna(0.0)
            feat[f"{column}_max{window}"] = roll.max()
    feat["requests_ratio_15"] = frame["requests"] / feat["requests_mean15"].replace(0, np.nan)
    feat["latency_ratio_60"] = frame["latency_p95"] / feat["latency_p95_mean60"].replace(0, np.nan)
    feat["error_delta_15"] = frame["error_rate"] - feat["error_rate_mean15"]
    feat["requests_diff1"] = frame["requests"].diff()
    feat["latency_diff1"] = frame["latency_p95"].diff()
    feat["bytes_per_request"] = frame["bytes_per_request"]
    feat["unique_ratio"] = frame["unique_users"] / frame["requests"].replace(0, np.nan)

    horizon = cfg["ml"]["horizon"]
    target_time = frame.index + pd.Timedelta(minutes=horizon)
    minute_of_day = target_time.hour.values * 60 + target_time.minute.values
    feat["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    feat["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    feat["dow"] = target_time.dayofweek.values
    feat["is_weekend"] = (target_time.dayofweek.values >= 5).astype(int)
    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def feature_columns(cfg):
    stub_index = pd.date_range("2026-01-01", periods=200, freq="1min")
    stub = pd.DataFrame(0.0, index=stub_index, columns=BASE_COLUMNS)
    return list(build_features(stub, cfg).columns)


LAG_COLUMNS = ["requests", "error_rate", "latency_p95", "latency_p50", "server_error_rate"]
ROLL_COLUMNS = ["requests", "latency_p95", "error_rate"]


class OnlineFeatureBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.depth = max(max(cfg["ml"]["lags"]), max(cfg["ml"]["roll_windows"])) + 5
        self.rows = []
        self.stamps = []
        self.series = {name: [] for name in BASE_COLUMNS}

    def push(self, minute, row):
        values = {name: float(row.get(name, 0.0)) for name in BASE_COLUMNS}
        self.rows.append(values)
        self.stamps.append(pd.Timestamp(minute))
        for name in BASE_COLUMNS:
            self.series[name].append(values[name])
            if len(self.series[name]) > self.depth:
                self.series[name].pop(0)
        if len(self.rows) > self.depth:
            self.rows.pop(0)
            self.stamps.pop(0)

    def latest(self):
        frame = pd.DataFrame(self.rows, index=pd.DatetimeIndex(self.stamps))
        return build_features(frame, self.cfg).iloc[[-1]]

    def latest_values(self):
        lags = self.cfg["ml"]["lags"]
        windows = self.cfg["ml"]["roll_windows"]
        arrays = {name: np.asarray(self.series[name], dtype=float) for name in BASE_COLUMNS}
        size = arrays["requests"].size
        out = {}

        for column in LAG_COLUMNS:
            data = arrays[column]
            for lag in lags:
                out[column + "_lag" + str(lag)] = data[-lag] if size >= lag else np.nan

        for column in ROLL_COLUMNS:
            data = arrays[column]
            for window in windows:
                tail = data[-window:]
                if tail.size >= 2:
                    out[column + "_mean" + str(window)] = float(tail.mean())
                    out[column + "_std" + str(window)] = float(tail.std(ddof=1))
                    out[column + "_max" + str(window)] = float(tail.max())
                else:
                    out[column + "_mean" + str(window)] = np.nan
                    out[column + "_std" + str(window)] = 0.0
                    out[column + "_max" + str(window)] = np.nan

        current = {name: arrays[name][-1] for name in BASE_COLUMNS}
        out["requests_ratio_15"] = safe_ratio(current["requests"], out["requests_mean15"])
        out["latency_ratio_60"] = safe_ratio(current["latency_p95"], out["latency_p95_mean60"])
        out["error_delta_15"] = current["error_rate"] - out["error_rate_mean15"]
        out["requests_diff1"] = (current["requests"] - arrays["requests"][-2]) if size >= 2 else np.nan
        out["latency_diff1"] = (current["latency_p95"] - arrays["latency_p95"][-2]) if size >= 2 else np.nan
        out["bytes_per_request"] = current["bytes_per_request"]
        out["unique_ratio"] = safe_ratio(current["unique_users"], current["requests"])

        target = self.stamps[-1] + pd.Timedelta(minutes=self.cfg["ml"]["horizon"])
        minute_of_day = target.hour * 60 + target.minute
        out["tod_sin"] = float(np.sin(2 * np.pi * minute_of_day / 1440.0))
        out["tod_cos"] = float(np.cos(2 * np.pi * minute_of_day / 1440.0))
        out["dow"] = float(target.dayofweek)
        out["is_weekend"] = float(target.dayofweek >= 5)
        return {key: clean(value) for key, value in out.items()}

    def latest_vector(self, columns):
        values = self.latest_values()
        return np.array([[values[name] for name in columns]], dtype=float)

    def ready(self):
        return len(self.rows) >= 3


def safe_ratio(numerator, denominator):
    if denominator is None or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def clean(value):
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)
