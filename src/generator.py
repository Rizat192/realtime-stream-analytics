import numpy as np
import pandas as pd

STATUS_OK = [200, 201, 204]
STATUS_OK_P = [0.86, 0.10, 0.04]
STATUS_CLIENT = [400, 401, 404, 429]
STATUS_CLIENT_P = [0.35, 0.25, 0.30, 0.10]
STATUS_SERVER = [500, 502, 503, 504]
STATUS_SERVER_P = [0.45, 0.20, 0.25, 0.10]


def seasonal_factor(minute_of_day, cfg):
    phase = 2.0 * np.pi * (minute_of_day - 300.0) / 1440.0
    primary = cfg["diurnal_amplitude"] * np.sin(phase)
    secondary = cfg["secondary_amplitude"] * np.sin(2.0 * phase)
    return np.clip(1.0 + primary + secondary, 0.15, None)


def build_anomaly_plan(cfg, minutes, rng):
    plan = np.zeros(minutes, dtype=np.int8)
    kinds = np.full(minutes, "none", dtype=object)
    placed = 0
    attempts = 0
    while placed < cfg["anomalies"]["count"] and attempts < 2000:
        attempts += 1
        duration = int(rng.integers(cfg["anomalies"]["min_duration"], cfg["anomalies"]["max_duration"] + 1))
        start = int(rng.integers(120, minutes - duration - 120))
        gap = cfg["anomalies"]["min_gap"]
        low = max(0, start - gap)
        high = min(minutes, start + duration + gap)
        if plan[low:high].any():
            continue
        kind = rng.choice(cfg["anomalies"]["types"], p=cfg["anomalies"]["type_weights"])
        plan[start:start + duration] = 1
        kinds[start:start + duration] = kind
        placed += 1
    return plan, kinds


def sample_status(n, error_rate, server_share, rng):
    if n == 0:
        return np.empty(0, dtype=np.int32)
    roll = rng.random(n)
    status = rng.choice(STATUS_OK, size=n, p=STATUS_OK_P).astype(np.int32)
    is_error = roll < error_rate
    n_err = int(is_error.sum())
    if n_err:
        shares = np.asarray(server_share, dtype=float)
        shares = shares[is_error] if shares.ndim else np.full(n_err, float(server_share))
        server_mask = rng.random(n_err) < shares
        errs = np.where(
            server_mask,
            rng.choice(STATUS_SERVER, size=n_err, p=STATUS_SERVER_P),
            rng.choice(STATUS_CLIENT, size=n_err, p=STATUS_CLIENT_P),
        )
        status[is_error] = errs.astype(np.int32)
    return status


def generate(cfg_root):
    cfg = cfg_root["generator"]
    rng = np.random.default_rng(cfg_root["seed"])
    minutes = int(cfg["minutes"])
    start = pd.Timestamp(cfg["start"])
    plan, kinds = build_anomaly_plan(cfg, minutes, rng)

    index = pd.date_range(start, periods=minutes, freq="1min")
    minute_of_day = index.hour.values * 60 + index.minute.values
    dow = index.dayofweek.values
    season = seasonal_factor(minute_of_day, cfg)
    week = np.where(dow >= 5, cfg["weekend_factor"], 1.0)
    drift = np.linspace(1.0, 1.12, minutes)
    noise = np.exp(rng.normal(0.0, cfg["noise_sigma"], minutes))
    intensity = season * week * drift * noise

    chunks = []
    for endpoint in cfg["endpoints"]:
        rate = endpoint["base_rate"] * intensity
        latency_scale = np.full(minutes, endpoint["base_latency"], dtype=float)
        error_rate = np.full(minutes, endpoint["error_rate"], dtype=float)
        server_share = np.full(minutes, 0.25, dtype=float)

        surge = kinds == "traffic_surge"
        drop = kinds == "traffic_drop"
        spike = kinds == "latency_spike"
        burst = kinds == "error_burst"
        rate = np.where(surge, rate * cfg["anomalies"]["surge_multiplier"], rate)
        rate = np.where(drop, rate * cfg["anomalies"]["drop_multiplier"], rate)
        latency_scale = np.where(spike, latency_scale * cfg["anomalies"]["latency_multiplier"], latency_scale)
        latency_scale = np.where(burst, latency_scale * 1.6, latency_scale)
        error_rate = np.where(burst, cfg["anomalies"]["error_rate_level"], error_rate)
        error_rate = np.where(spike, error_rate * 3.0, error_rate)
        server_share = np.where(burst, 0.85, server_share)
        server_share = np.where(spike, 0.60, server_share)

        counts = rng.poisson(np.clip(rate, 0.05, None))
        total = int(counts.sum())
        if total == 0:
            continue
        minute_idx = np.repeat(np.arange(minutes), counts)
        offsets = rng.random(total) * 60.0
        timestamps = index.values[minute_idx] + (offsets * 1e9).astype("timedelta64[ns]")
        lat = rng.lognormal(np.log(latency_scale[minute_idx]), endpoint["latency_sigma"])
        status = sample_status(total, error_rate[minute_idx], server_share[minute_idx], rng)
        lat = np.where(status >= 500, lat * 1.8, lat)
        region = rng.choice(cfg["regions"], size=total, p=cfg["region_weights"])
        payload = rng.lognormal(np.log(2400.0), 0.7, total)
        payload = np.where(status >= 400, payload * 0.2, payload)

        chunks.append(pd.DataFrame({
            "timestamp": timestamps,
            "endpoint": endpoint["name"],
            "region": region,
            "latency_ms": np.round(lat, 2),
            "status": status,
            "bytes_out": np.round(payload).astype(np.int32),
            "user_id": rng.integers(1, 25000, total),
        }))

    events = pd.concat(chunks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    truth = pd.DataFrame({
        "minute": index,
        "is_anomaly": plan.astype(np.int8),
        "anomaly_type": kinds,
    })
    return events, truth
