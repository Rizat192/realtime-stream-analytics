import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SIGNALS = ["requests", "latency_p95", "error_rate"]
IFOREST_FEATURES = [
    "requests",
    "latency_p50",
    "latency_p95",
    "error_rate",
    "server_error_rate",
    "bytes_per_request",
    "requests_ratio",
    "latency_ratio",
]

SCALE = 1.4826
EPS = 1e-9


def robust_scores(minutes, cfg):
    window = cfg["anomaly"]["zscore_window"]
    min_periods = cfg["anomaly"]["zscore_min_periods"]
    scores = pd.DataFrame(index=minutes.index)
    for name in SIGNALS:
        series = minutes[name].astype(float)
        past = series.shift(1)
        center = past.rolling(window, min_periods=min_periods).median()
        deviation = (past - center).abs()
        spread = deviation.rolling(window, min_periods=min_periods).median() * SCALE
        scores[name] = ((series - center) / (spread + EPS)).abs()
    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scores["score"] = scores[SIGNALS].max(axis=1)
    return scores


def iforest_frame(minutes, cfg):
    window = cfg["timeseries"]["rolling_window"]
    frame = minutes.copy()
    frame["requests_ratio"] = frame["requests"] / frame["requests"].rolling(window, min_periods=1).mean().shift(1)
    frame["latency_ratio"] = frame["latency_p95"] / frame["latency_p95"].rolling(window, min_periods=1).mean().shift(1)
    return frame[IFOREST_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(1.0)


def fit_iforest(train_frame, cfg):
    model = IsolationForest(
        n_estimators=cfg["anomaly"]["iforest_estimators"],
        contamination=cfg["anomaly"]["iforest_contamination"],
        random_state=cfg["seed"],
        n_jobs=1,
    )
    model.fit(train_frame.values)
    return model


def evaluate(y_true, y_score, y_pred, name):
    return {
        "method": name,
        "alarms": int(np.sum(y_pred)),
        "positives": int(np.sum(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def event_recall(truth, predictions):
    labels = truth["is_anomaly"].values
    flags = np.asarray(predictions, dtype=int)
    events = 0
    caught = 0
    index = 0
    while index < len(labels):
        if labels[index] == 1:
            end = index
            while end < len(labels) and labels[end] == 1:
                end += 1
            events += 1
            if flags[index:end].any():
                caught += 1
            index = end
        else:
            index += 1
    return {"anomaly_events": events, "events_detected": caught,
            "event_recall": float(caught / events) if events else 0.0}


class OnlineRobustDetector:
    def __init__(self, cfg):
        self.window = cfg["anomaly"]["zscore_window"]
        self.min_periods = cfg["anomaly"]["zscore_min_periods"]
        self.threshold = cfg["anomaly"]["zscore_threshold"]
        self.history = {name: [] for name in SIGNALS}

    def update(self, row):
        score = 0.0
        for name in SIGNALS:
            value = float(row[name])
            past = self.history[name]
            if len(past) >= self.min_periods:
                array = np.asarray(past[-self.window:], dtype=float)
                center = np.median(array)
                spread = np.median(np.abs(array - center)) * SCALE
                current = abs(value - center) / (spread + EPS)
                score = max(score, float(current))
            past.append(value)
            if len(past) > self.window:
                past.pop(0)
        return score, int(score > self.threshold)
