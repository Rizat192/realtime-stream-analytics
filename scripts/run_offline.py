import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import anomaly, ml, plots, timeseries
from src.config import load_config, path_in


def load_minutes(cfg):
    minutes = pd.read_csv(path_in(cfg, "data", "minutes.csv"), parse_dates=["minute"])
    minutes = minutes.set_index("minute")
    truth = pd.read_csv(path_in(cfg, "data", "labels.csv"), parse_dates=["minute"])
    return minutes, truth


def run_timeseries(cfg, minutes, truth):
    view = timeseries.rolling_view(minutes, cfg)
    parts = timeseries.decompose(minutes, cfg)
    hourly, by_hour, by_dow = timeseries.resample_profiles(minutes)
    summary = timeseries.summarize(minutes, parts, cfg)
    plots.timeseries_overview(view, truth, path_in(cfg, "figures", "fig_timeseries_overview.png"))
    plots.stl_figure(parts, path_in(cfg, "figures", "fig_stl_decomposition.png"))
    plots.seasonal_profiles(by_hour, by_dow, hourly, path_in(cfg, "figures", "fig_seasonal_profiles.png"))
    parts.to_csv(path_in(cfg, "results", "stl_components.csv"))
    return summary


def run_anomaly(cfg, minutes, truth, cut):
    robust = anomaly.robust_scores(minutes, cfg)
    threshold = cfg["anomaly"]["zscore_threshold"]
    frame = anomaly.iforest_frame(minutes, cfg)
    model = anomaly.fit_iforest(frame.iloc[:cut], cfg)
    iforest_score = -model.score_samples(frame.values)
    iforest_flag = (model.predict(frame.values) == -1).astype(int)

    results = pd.DataFrame({
        "robust_score": robust["score"].values,
        "robust_flag": (robust["score"] > threshold).astype(int).values,
        "iforest_score": iforest_score,
        "iforest_flag": iforest_flag,
    }, index=minutes.index)

    labels = truth.set_index("minute")["is_anomaly"].reindex(minutes.index).fillna(0).astype(int)
    test = slice(cut, len(minutes))
    y_true = labels.values[test]
    metrics = [
        anomaly.evaluate(y_true, results["robust_score"].values[test],
                         results["robust_flag"].values[test], "rolling robust z score"),
        anomaly.evaluate(y_true, results["iforest_score"].values[test],
                         results["iforest_flag"].values[test], "isolation forest"),
    ]
    metrics[0].update(anomaly.event_recall(truth.iloc[cut:], results["robust_flag"].values[test]))
    metrics[1].update(anomaly.event_recall(truth.iloc[cut:], results["iforest_flag"].values[test]))

    plots.anomaly_figure(minutes, truth, results, threshold,
                         path_in(cfg, "figures", "fig_anomaly_detection.png"))
    plots.curves_figure(
        [("rolling robust z score", y_true, results["robust_score"].values[test], plots.BLUE),
         ("isolation forest", y_true, results["iforest_score"].values[test], plots.GREEN)],
        path_in(cfg, "figures", "fig_anomaly_curves.png"),
        "Detector comparison on the held out window",
    )
    results.to_csv(path_in(cfg, "results", "anomaly_scores.csv"))
    joblib.dump({"model": model, "columns": list(frame.columns)},
                path_in(cfg, "models", "iforest.joblib"))
    return metrics, results


def run_ml(cfg, minutes, truth, anomaly_results):
    x, y_clf = ml.build_matrix(minutes, truth, cfg)
    cut = ml.split_index(len(x), cfg["ml"]["test_fraction"])
    x_train, x_test = x.iloc[:cut], x.iloc[cut:]
    clf_train, clf_test = y_clf.iloc[:cut], y_clf.iloc[cut:]

    clf = ml.train_classifier(x_train, clf_train, cfg)
    proba = clf.predict_proba(x_test.values)[:, 1]
    tuned, tuned_f1 = ml.best_threshold(clf_test.values, proba)

    detector_score = anomaly_results["robust_score"].reindex(x_test.index).fillna(0.0).values
    detector_flag = anomaly_results["robust_flag"].reindex(x_test.index).fillna(0).values
    clf_metrics = [
        ml.evaluate_classifier(clf_test.values, proba, cfg["ml"]["classifier_threshold"],
                               "gradient boosting, default threshold"),
        ml.evaluate_classifier(clf_test.values, proba, tuned, "gradient boosting, tuned threshold"),
        ml.evaluate_classifier(clf_test.values, detector_score,
                               cfg["anomaly"]["zscore_threshold"],
                               "current detector forwarded as prediction"),
    ]
    onset = ml.onset_report({
        "gradient boosting, tuned threshold": (proba >= tuned).astype(int),
        "current detector forwarded as prediction": detector_flag.astype(int),
    }, clf_test.values)

    importance = ml.permutation_importance_fast(clf, x_test, clf_test, cfg)
    plots.classification_timeline(x_test.index, proba, clf_test.values, tuned,
                                  path_in(cfg, "figures", "fig_ml_classification.png"))
    plots.importance_figure(importance, path_in(cfg, "figures", "fig_feature_importance.png"))

    joblib.dump({
        "classifier": clf,
        "columns": list(x.columns),
        "threshold": tuned,
    }, path_in(cfg, "models", "stream_models.joblib"))

    predictions = pd.DataFrame({
        "incident_proba": proba,
        "incident_true": clf_test.values,
    }, index=x_test.index)
    predictions.to_csv(path_in(cfg, "results", "ml_predictions.csv"))
    importance.to_csv(path_in(cfg, "results", "feature_importance.csv"), index=False)

    info = {
        "rows_total": int(len(x)),
        "features": int(x.shape[1]),
        "train_rows": int(cut),
        "test_rows": int(len(x) - cut),
        "train_end": str(x_train.index[-1]),
        "test_start": str(x_test.index[0]),
        "positive_rate_train": float(clf_train.mean()),
        "positive_rate_test": float(clf_test.mean()),
        "tuned_threshold": tuned,
        "tuned_threshold_f1": tuned_f1,
    }
    return info, clf_metrics, importance, onset


def main():
    cfg = load_config()
    minutes, truth = load_minutes(cfg)
    cut = ml.split_index(len(minutes), cfg["ml"]["test_fraction"])

    ts_summary = run_timeseries(cfg, minutes, truth)
    anomaly_metrics, anomaly_results = run_anomaly(cfg, minutes, truth, cut)
    ml_info, clf_metrics, importance, onset = run_ml(cfg, minutes, truth, anomaly_results)
    plots.architecture_figure(cfg, path_in(cfg, "figures", "fig_architecture.png"))

    dataset = {
        "minute_windows": int(len(minutes)),
        "anomalous_minutes": int(truth["is_anomaly"].sum()),
        "anomaly_rate": float(truth["is_anomaly"].mean()),
        "anomaly_types": {k: int(v) for k, v in
                          truth.loc[truth["is_anomaly"] == 1, "anomaly_type"].value_counts().items()},
        "mean_requests_per_minute": float(minutes["requests"].mean()),
        "mean_latency_p95": float(minutes["latency_p95"].mean()),
        "mean_error_rate": float(minutes["error_rate"].mean()),
    }
    payload = {
        "dataset": dataset,
        "timeseries": ts_summary,
        "anomaly": anomaly_metrics,
        "ml_info": ml_info,
        "classification": clf_metrics,
        "onset": onset,
        "importance": importance.to_dict("records"),
    }
    out = path_in(cfg, "results", "offline_results.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps({"timeseries": ts_summary}, indent=2))
    print(json.dumps({"anomaly": anomaly_metrics}, indent=2))
    print(json.dumps({"classification": clf_metrics}, indent=2))
    print(json.dumps({"onset": onset}, indent=2))
    print("saved " + out)


if __name__ == "__main__":
    main()
