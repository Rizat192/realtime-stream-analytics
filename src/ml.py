import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .features import build_features


def make_targets(minutes, truth, cfg):
    horizon = cfg["ml"]["horizon"]
    labels = truth.set_index("minute")["is_anomaly"].reindex(minutes.index).fillna(0).astype(int)
    return labels.shift(-horizon)


def split_index(n, test_fraction):
    cut = int(n * (1.0 - test_fraction))
    return cut


def build_matrix(minutes, truth, cfg):
    features = build_features(minutes, cfg)
    y_clf = make_targets(minutes, truth, cfg)
    warmup = max(cfg["ml"]["roll_windows"]) + max(cfg["ml"]["lags"])
    mask = y_clf.notna()
    mask.iloc[:warmup] = False
    return features[mask], y_clf[mask].astype(int)


def train_classifier(x_train, y_train, cfg):
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=6,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=cfg["seed"],
    )
    model.fit(x_train.values, y_train.values)
    return model


def evaluate_classifier(y_true, proba, threshold, name):
    pred = (proba >= threshold).astype(int)
    return {
        "model": name,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "positives": int(np.sum(y_true)),
        "predicted_positives": int(pred.sum()),
    }


def best_threshold(y_true, proba):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    index = int(np.argmax(scores))
    return float(grid[index]), float(scores[index])


def permutation_importance_fast(model, x_test, y_test, cfg, top=12):
    rng = np.random.default_rng(cfg["seed"])
    base = roc_auc_score(y_test, model.predict_proba(x_test.values)[:, 1])
    drops = {}
    for column in x_test.columns:
        shuffled = x_test.copy()
        shuffled[column] = rng.permutation(shuffled[column].values)
        score = roc_auc_score(y_test, model.predict_proba(shuffled.values)[:, 1])
        drops[column] = float(base - score)
    ranked = sorted(drops.items(), key=lambda item: item[1], reverse=True)[:top]
    return pd.DataFrame(ranked, columns=["feature", "roc_auc_drop"])


def onset_report(predictions, y_true):
    labels = np.asarray(y_true, dtype=int)
    previous = np.concatenate([[0], labels[:-1]])
    onset = (labels == 1) & (previous == 0)
    ongoing = (labels == 1) & (previous == 1)
    rows = []
    for name, pred in predictions.items():
        pred = np.asarray(pred, dtype=int)
        rows.append({
            "model": name,
            "onset_minutes": int(onset.sum()),
            "onset_recall": float(pred[onset].mean()) if onset.any() else 0.0,
            "ongoing_recall": float(pred[ongoing].mean()) if ongoing.any() else 0.0,
        })
    return rows
