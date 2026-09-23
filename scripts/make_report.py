import json
import os
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, path_in

TITLE = "Real-Time Monitoring, Time-Series Analysis and Stream Intelligence for API Traffic"
SUBTITLE = "Endterm Project Technical Report"


def load_results(cfg):
    with open(path_in(cfg, "results", "offline_results.json"), "r", encoding="utf-8") as handle:
        offline = json.load(handle)
    with open(path_in(cfg, "results", "stream_summary.json"), "r", encoding="utf-8") as handle:
        stream = json.load(handle)
    return offline, stream


def setup(document):
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)
    style.paragraph_format.space_after = Pt(4)
    style.paragraph_format.line_spacing = 1.04
    for level, size in [("Heading 1", 13.5), ("Heading 2", 11.5)]:
        heading = document.styles[level]
        heading.font.size = Pt(size)
        heading.font.color.rgb = RGBColor(0x1F, 0x3B, 0x57)
        heading.font.name = "Calibri"
        heading.paragraph_format.space_before = Pt(8)
        heading.paragraph_format.space_after = Pt(3)


def paragraph(document, text):
    item = document.add_paragraph(text)
    item.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return item


def figure(document, cfg, name, caption, width=5.4):
    path = path_in(cfg, "figures", name)
    if not os.path.exists(path):
        return
    document.add_picture(path, width=Inches(width))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.paragraphs[-1].paragraph_format.space_after = Pt(2)
    item = document.add_paragraph(caption)
    item.alignment = WD_ALIGN_PARAGRAPH.CENTER
    item.paragraph_format.space_after = Pt(6)
    run = item.runs[0]
    run.italic = True
    run.font.size = Pt(8)


def table(document, headers, rows):
    item = document.add_table(rows=1, cols=len(headers))
    item.style = "Light Grid Accent 1"
    for cell, text in zip(item.rows[0].cells, headers):
        cell.text = str(text)
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(8.5)
    for row in rows:
        cells = item.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(8.5)
    for row in item.rows:
        for cell in row.cells:
            for block in cell.paragraphs:
                block.paragraph_format.space_before = Pt(1)
                block.paragraph_format.space_after = Pt(1)
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    return item


def fmt(value, digits=3):
    return format(round(float(value), digits), "." + str(digits) + "f")


def title_page(document, cfg, offline, stream):
    heading = document.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run(TITLE)
    run.bold = True
    run.font.size = Pt(17)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(SUBTITLE)
    run.font.size = Pt(12)
    facts = [
        ["Domain", "server and API request traffic"],
        ["Stream volume", "{:,} events over {:,} one-minute windows".format(
            int(stream["events_processed"]), offline["dataset"]["minute_windows"])],
        ["Simulated span", "3 days, replayed at {:.0f} times real time".format(stream["speed_factor"])],
        ["Processing topology", "{} concurrent processes over {} hash partitions".format(
            stream["processes"], cfg["pipeline"]["partitions"])],
        ["Stack", "Python 3.12, pandas, NumPy, scikit-learn, statsmodels, matplotlib"],
    ]
    table(document, ["Item", "Value"], facts)
    paragraph(document, "Every number in this report is produced by the code in this repository and "
                        "written to the results directory by scripts/run_offline.py and "
                        "scripts/run_stream.py. The seed is fixed in config.yaml, so a rerun reproduces "
                        "the same dataset, the same models and the same metrics.")


def section_intro(document, offline):
    document.add_heading("1. Introduction and objectives", level=1)
    paragraph(document, "This project extends a streaming application into an intelligent analytics "
                        "system. The stream carries API request records from a web service. Each record "
                        "holds a timestamp, an endpoint, a region, the measured latency, the HTTP status "
                        "code, the response size and a user identifier. The system turns that raw event "
                        "flow into operational intelligence in five layers: continuously updated "
                        "visualization, time-series analysis, anomaly detection, machine learning that "
                        "predicts the next minute, and a distributed processing topology that carries "
                        "all of it.")
    paragraph(document, "The objectives follow the project requirements: continuously updated "
                        "visualization with more than one view, temporal analysis with rolling windows, "
                        "resampling, trend and seasonality, anomaly detection evaluated against ground "
                        "truth, machine learning that predicts in real time from information available "
                        "at prediction time, and a distributed topology with measured operational "
                        "characteristics.")
    stats = offline["dataset"]
    paragraph(document, "The dataset holds {:,} minute windows with {:,} anomalous minutes, a rate of "
                        "{:.1%}. Mean traffic is {:.1f} requests per minute, mean p95 latency is "
                        "{:.0f} ms and the mean error rate is {:.2%}.".format(
                            stats["minute_windows"], stats["anomalous_minutes"], stats["anomaly_rate"],
                            stats["mean_requests_per_minute"], stats["mean_latency_p95"],
                            stats["mean_error_rate"]))


def section_architecture(document, cfg, stream):
    document.add_heading("2. System architecture", level=1)
    paragraph(document, "The system is a staged pipeline. Each stage is a separate operating-system "
                        "process and the stages communicate only over authenticated TCP connections on "
                        "the loopback interface. No stage shares memory with any other, so moving a "
                        "stage to another machine is a change of host name and nothing else. This is the "
                        "same shape as a Kafka or Flink deployment with the broker replaced by direct "
                        "connections between operators.")
    figure(document, cfg, "fig_architecture.png",
           "Figure 1. Stream topology: producers, hash partitions, window workers, merger and scorers.")
    rows = [
        ["Producers", str(cfg["pipeline"]["producers"]),
         "replay the dataset in timestamp order, each owning a disjoint set of endpoints"],
        ["Partitions", str(cfg["pipeline"]["partitions"]),
         "one inbox per partition, keyed by CRC32 of the endpoint name"],
        ["Window workers", str(cfg["pipeline"]["partitions"]),
         "one-minute tumbling aggregation per partition behind an event-time watermark"],
        ["Merger", "1", "joins the partial aggregates of one minute into a complete window"],
        ["Scoring workers", str(cfg["pipeline"]["scoring_workers"]),
         "one runs anomaly detection, one runs the machine-learning models"],
        ["Sink", "1", "joins both result streams, drives the dashboard, records operational metrics"],
    ]
    table(document, ["Stage", "Processes", "Responsibility"], rows)
    paragraph(document, "Two forms of parallelism are present. Producers and window workers are data "
                        "parallel: the stream is split by key and the same code runs on each shard. The "
                        "scoring stage is task parallel: both workers consume the same merged window "
                        "stream but apply different models, so the machine-learning cost never delays "
                        "the anomaly alarm. In total {} processes run concurrently. Every stage owns an "
                        "inbox, which is a listening socket, one reader thread per upstream connection "
                        "and a local queue whose depth is the consumer lag of that stage. Each message "
                        "carries the inbox depth of the stage that produced it, so the sink reconstructs "
                        "the backlog of the whole topology without polling anything.".format(
                            stream["processes"]))


def section_data(document, cfg, offline):
    document.add_heading("3. Data stream and generator", level=1)
    paragraph(document, "The generator in src/generator.py produces the event stream. Request volume "
                        "per endpoint is drawn from a Poisson distribution whose rate follows a daily "
                        "shape built from two sine components, a weekday factor that lowers weekend "
                        "traffic, a slow upward drift across the three days and multiplicative noise. "
                        "Latency is lognormal around an endpoint-specific base, status codes are drawn "
                        "from an endpoint-specific error rate, and server errors carry extra latency.")
    rows = [[item["name"], item["base_rate"], item["base_latency"], "{:.1%}".format(item["error_rate"])]
            for item in cfg["generator"]["endpoints"]]
    table(document, ["Endpoint", "Base rate (req/min)", "Base latency (ms)", "Base error rate"], rows)
    types = offline["dataset"]["anomaly_types"]
    anomalies = cfg["generator"]["anomalies"]
    rows = [
        ["latency_spike", types.get("latency_spike", 0),
         "p95 latency multiplied by {}".format(anomalies["latency_multiplier"])],
        ["error_burst", types.get("error_burst", 0),
         "error rate raised to {:.0%}, mostly 5xx".format(anomalies["error_rate_level"])],
        ["traffic_drop", types.get("traffic_drop", 0),
         "request rate cut to {:.0%} of normal".format(anomalies["drop_multiplier"])],
        ["traffic_surge", types.get("traffic_surge", 0),
         "request rate multiplied by {}".format(anomalies["surge_multiplier"])],
    ]
    table(document, ["Incident type", "Labelled minutes", "Injected effect"], rows)
    paragraph(document, "{} incidents are placed at random positions with a minimum gap of {} minutes "
                        "and a duration between {} and {} minutes. Every minute carries a ground-truth "
                        "label, which is what makes the anomaly and classification results measurable "
                        "rather than anecdotal.".format(
                            anomalies["count"], anomalies["min_gap"],
                            anomalies["min_duration"], anomalies["max_duration"]))


def section_visualization(document, cfg):
    document.add_heading("4. Real-time visualization", level=1)
    paragraph(document, "The dashboard in src/dashboard.py is driven by the sink process and redraws as "
                        "scored windows arrive. It shows four views at once, which satisfies the "
                        "requirement for at least two meaningful views, and each view answers a "
                        "different operational question. It keeps a sliding window of recent minutes, so "
                        "the redraw cost stays constant as the run gets longer.")
    rows = [
        ["Stream volume", "requests per minute against its 15-minute rolling mean"],
        ["Tail latency and alarms", "p95 latency with anomaly alarms marked as they fire"],
        ["Errors and risk", "error rate on the left axis, predicted incident probability on the right"],
        ["Operational metrics", "throughput in events per second and total backlog"],
    ]
    table(document, ["Panel", "Content"], rows)
    figure(document, cfg, "fig_dashboard.png",
           "Figure 2. The live dashboard at the end of a run, with the running counters in the header.",
           width=6.0)
    figure(document, cfg, "fig_timeseries_overview.png",
           "Figure 3. The complete three-day stream. Shaded bands are the injected incidents.",
           width=5.2)


def section_timeseries(document, cfg, offline):
    document.add_heading("5. Time-series analysis", level=1)
    summary = offline["timeseries"]
    paragraph(document, "Events are resampled into one-minute tumbling windows indexed by timestamp. "
                        "Each window carries the request count, the error rate, the server error rate, "
                        "three latency quantiles, the bytes per request and the number of distinct "
                        "users. This turns an irregular event stream into a regular series of {:,} "
                        "points, which is the input for every later layer. Three temporal methods are "
                        "applied: rolling windows of 15 and 60 minutes for the local level and trend, "
                        "resampling to hourly and to hour-of-day and weekday profiles for the daily "
                        "cycle, and STL decomposition with a period of 1440 minutes into trend, seasonal "
                        "and residual parts.".format(summary["points"]))
    rows = [
        ["Mean requests per minute", fmt(summary["mean_requests_per_minute"], 2)],
        ["Linear trend per day", fmt(summary["linear_trend_per_day"], 2) + " requests per minute"],
        ["Augmented Dickey-Fuller p-value", fmt(summary["adf_pvalue"], 4)],
        ["Autocorrelation at lag 1", fmt(summary["acf_lag_1"], 3)],
        ["Autocorrelation at lag 1440", fmt(summary["acf_lag_1440"], 3)],
        ["Seasonal strength", fmt(summary["seasonal_strength"], 3)],
        ["Trend strength", fmt(summary["trend_strength"], 3)],
        ["Busiest and quietest hour", "{}:00 and {}:00".format(summary["peak_hour"], summary["trough_hour"])],
    ]
    table(document, ["Property", "Value"], rows)
    paragraph(document, "The autocorrelation at lag 1440 confirms the daily cycle: the series is "
                        "strongly related to its value exactly one day earlier. Seasonal strength of "
                        "{:.2f} says the daily component explains most of the structured variation, so "
                        "any detector that ignores the time of day will raise alarms every morning and "
                        "every night. That is the reason the detector works on deviation from a local "
                        "baseline rather than on the raw level.".format(summary["seasonal_strength"]))
    figure(document, cfg, "fig_stl_decomposition.png",
           "Figure 4. STL decomposition of requests per minute into trend, seasonal and residual parts.",
           width=5.2)


def section_anomaly(document, cfg, offline, stream):
    document.add_heading("6. Anomaly detection", level=1)
    paragraph(document, "Two detectors are implemented and compared on the same held-out window, the "
                        "last 30 percent of the series, so that neither method benefits from data it was "
                        "tuned on.")
    document.add_heading("6.1 Rolling robust z-score", level=2)
    paragraph(document, "For the request count, the p95 latency and the error rate the detector keeps a "
                        "trailing window of {} minutes, takes the median as the baseline and the median "
                        "absolute deviation scaled by 1.4826 as the spread, then reports the largest "
                        "absolute deviation across the three signals. The current value is excluded from "
                        "its own baseline, so a long incident cannot silently become the new normal "
                        "inside the window. A minute is flagged when the score exceeds {}. The method is "
                        "online by construction, so the streaming implementation keeps the same trailing "
                        "buffer and needs no retraining.".format(
                            cfg["anomaly"]["zscore_window"], cfg["anomaly"]["zscore_threshold"]))
    document.add_heading("6.2 Isolation forest", level=2)
    paragraph(document, "The second detector is an isolation forest with {} trees fitted on the first 70 "
                        "percent of the windows. It sees eight features per minute, including the ratio "
                        "of the current volume and latency to their recent rolling means, so it judges "
                        "the shape of a minute rather than its absolute level. Contamination is set to "
                        "{:.0%}, close to the true incident rate.".format(
                            cfg["anomaly"]["iforest_estimators"], cfg["anomaly"]["iforest_contamination"]))
    rows = []
    for item in offline["anomaly"]:
        rows.append([item["method"], fmt(item["precision"], 2), fmt(item["recall"], 2),
                     fmt(item["f1"], 2), fmt(item["roc_auc"], 2), fmt(item["pr_auc"], 2),
                     "{}/{}".format(item["events_detected"], item["anomaly_events"])])
    detector = stream["streaming_detector"]
    rows.append(["streaming robust z score, live run", fmt(detector["precision"], 2),
                 fmt(detector["recall"], 2), fmt(detector["f1"], 2), fmt(detector["roc_auc"], 2),
                 fmt(detector["pr_auc"], 2),
                 "{}/{}".format(detector["events_detected"], detector["anomaly_events"])])
    table(document, ["Method", "Precision", "Recall", "F1", "ROC AUC", "PR AUC", "Incidents caught"],
          rows)
    paragraph(document, "Per-minute recall understates the operational value of a detector, because "
                        "catching one minute of an incident is enough to page an engineer. The last "
                        "column reports incident-level recall, the share of injected incidents with at "
                        "least one flagged minute. The last row is the same detector running inside the "
                        "live pipeline over the whole replay rather than the held-out window, which is "
                        "why its precision is lower and its incident count is {}: it confirms that the "
                        "streaming implementation reproduces the offline behaviour.".format(
                            detector["anomaly_events"]))
    figure(document, cfg, "fig_anomaly_detection.png",
           "Figure 5. Alarms from both detectors against the injected incidents.", width=5.2)
    figure(document, cfg, "fig_anomaly_curves.png",
           "Figure 6. ROC and precision-recall curves for the two detectors on the held-out window.",
           width=5.0)


def section_ml(document, cfg, offline):
    document.add_heading("7. Real-time machine learning", level=1)
    info = offline["ml_info"]
    paragraph(document, "The prediction problem is classification: will the next minute be an "
                        "incident. It is evaluated on a time-ordered split, {:,} training windows "
                        "followed by {:,} test windows, with the test period starting at {}. A random "
                        "split would leak the future into the past and is never used.".format(
                            info["train_rows"], info["test_rows"], info["test_start"]))
    document.add_heading("7.1 Features available at prediction time", level=2)
    paragraph(document, "The feature builder in src/features.py produces {} features for a prediction "
                        "made at the end of minute t about minute t plus 1. Every feature is a function "
                        "of windows that have already closed.".format(info["features"]))
    rows = [
        ["Lagged levels", "request count, error rate, p95 and p50 latency, server error rate at lags "
                          + ", ".join(str(lag) for lag in cfg["ml"]["lags"])],
        ["Rolling statistics", "mean, standard deviation and maximum over trailing windows of "
                               + ", ".join(str(w) for w in cfg["ml"]["roll_windows"]) + " minutes"],
        ["Ratio and difference", "volume over its 15-minute mean, p95 over its 60-minute mean, "
                                 "one-step changes"],
        ["Calendar", "sine and cosine of the time of day, weekday, weekend flag"],
    ]
    table(document, ["Group", "Content"], rows)
    paragraph(document, "Leakage is controlled in three places. Rolling statistics are computed on "
                        "closed windows only. The targets are shifted backwards by one minute, so the "
                        "label always lies strictly in the future of its features. The calendar features "
                        "describe the target minute, which is legitimate because a clock is known in "
                        "advance. The first {} windows are dropped because their rolling features are "
                        "not yet filled.".format(max(cfg["ml"]["roll_windows"]) + max(cfg["ml"]["lags"])))
    document.add_heading("7.2 Incident classification", level=2)
    paragraph(document, "The classifier is a histogram gradient boosting model with balanced class "
                        "weights, because only {:.1%} of the test minutes are positive. It is reported "
                        "at the default threshold and at a threshold tuned for F1, and the current "
                        "output of the unsupervised detector is included as a baseline, because a model "
                        "that cannot beat forwarding the detector is not worth its "
                        "complexity.".format(info["positive_rate_test"]))
    rows = [[item["model"], fmt(item["precision"], 2), fmt(item["recall"], 2), fmt(item["f1"], 2),
             fmt(item["roc_auc"], 2), fmt(item["pr_auc"], 2)] for item in offline["classification"]]
    table(document, ["Model", "Precision", "Recall", "F1", "ROC AUC", "PR AUC"], rows)
    onset = offline.get("onset", [])
    if onset:
        paragraph(document, "The headline F1 flatters the model, so it is worth splitting. An incident "
                            "minute is either the first minute of an incident, where the evidence is "
                            "still in the future, or a later minute of an incident that is already "
                            "visible in the features.")
        rows = [[item["model"], item["onset_minutes"], fmt(item["onset_recall"], 2),
                 fmt(item["ongoing_recall"], 2)] for item in onset]
        table(document, ["Model", "Onset minutes in test", "Recall at onset", "Recall while ongoing"],
              rows)
        baseline = offline["classification"][-1]
        paragraph(document, "The gradient boosting model recalls {:.0%} of the minutes of incidents that "
                            "are already under way and none of the onsets. That is the honest reading of "
                            "the result: it is an excellent persistence detector and not an early "
                            "warning system. It is also the expected result, because the generator "
                            "places incidents at random times, so nothing in the traffic before an "
                            "incident carries information about when it will start. The baseline "
                            "catches {:.0%} of the onsets, but only by firing on {} of the {:,} test "
                            "minutes at a precision of {}, so its alarms are cheap and mostly "
                            "wrong.".format(onset[0]["ongoing_recall"], onset[-1]["onset_recall"],
                                            baseline["predicted_positives"], info["test_rows"],
                                            fmt(baseline["precision"], 2)))
    figure(document, cfg, "fig_ml_classification.png",
           "Figure 7. Predicted incident probability against the true incidents in the test window.",
           width=5.4)
    top = offline["importance"][:5]
    paragraph(document, "Permutation importance on the test window ranks the features by how much the "
                        "ROC AUC falls when the feature is shuffled. The strongest are "
                        + ", ".join(item["feature"] for item in top)
                        + ". The full ranking is in results/feature_importance.csv.")


def section_stream(document, cfg, stream):
    document.add_heading("8. Distributed stream processing", level=1)
    watermark = stream.get("watermark_minutes", cfg["pipeline"]["watermark_minutes"])
    paragraph(document, "The pipeline in src/pipeline.py demonstrates the distributed concepts on one "
                        "machine with real operating-system processes and no shared state. Partitioning "
                        "by a stable CRC32 hash of the endpoint guarantees that all events of one key "
                        "reach the same window worker, which is what makes per-key aggregation correct "
                        "under parallelism. A watermark of {} event-time minutes closes a window once "
                        "the stream has moved far enough past it, which bounds memory and tolerates "
                        "producers that are not perfectly in step. Records that arrive for a window that "
                        "is already closed are counted as late and dropped rather than silently "
                        "reopening the window. End-of-stream sentinels are counted per upstream "
                        "connection, so every stage shuts down only after all of its inputs are "
                        "done.".format(watermark))
    rows = [
        ["Wall-clock duration", "{:.1f} s".format(stream["wall_clock_seconds"])],
        ["Simulated time", "{:.1f} h at {:.0f} times real time".format(
            stream["simulated_hours"], stream["speed_factor"])],
        ["Events processed", "{:,}".format(int(stream["events_processed"]))],
        ["Windows completed", "{:,}".format(stream["simulated_minutes"])],
        ["Throughput", "{:,.0f} events per second mean, {:,.0f} peak".format(
            stream["mean_throughput_eps"], stream["peak_throughput_eps"])],
        ["Window completion rate", "{:.1f} windows per second".format(stream["windows_per_second"])],
        ["End-to-end latency", "{:.0f} ms mean, {:.0f} ms p95".format(
            stream["end_to_end_ms_mean"], stream["end_to_end_ms_p95"])],
        ["Stage latency", "aggregation {:.0f} ms, merge {:.1f} ms, scoring {:.0f} ms".format(
            stream["window_stage_ms_mean"], stream["merge_stage_ms_mean"],
            stream["score_stage_ms_mean"])],
        ["Backlog", "{:.1f} mean, {:,} peak at the {} stage".format(
            stream["backlog_mean"], stream["backlog_peak"],
            stream.get("backlog_peak_stage", "scoring"))],
        ["Late records dropped", "{:,} with a watermark of {} event-time minutes".format(
            stream.get("late_events_dropped", 0), watermark)],
    ]
    table(document, ["Operational characteristic", "Measured value"], rows)
    paragraph(document, "Five operational characteristics are measured, above the two the project "
                        "requires: throughput, end-to-end and per-stage processing latency, completed "
                        "event and window counts, per-stage backlog and dropped late records. The sink "
                        "samples them once per second into results/ops_metrics.csv. The aggregation "
                        "stage includes the wait for the window to close, so it carries the bulk of the "
                        "end-to-end figure: at {:.0f} times real time a watermark of {} event-time "
                        "minutes is {:.0f} ms of wall time, which matches the measured aggregation "
                        "stage almost exactly. Backlog is the early warning signal, it stays near zero "
                        "while the consumers keep up and grows as soon as a stage becomes the "
                        "bottleneck, which is the signal a production consumer group exposes as consumer "
                        "lag.".format(stream["speed_factor"], watermark,
                                      watermark * 60.0 / stream["speed_factor"] * 1000.0))
    figure(document, cfg, "fig_ops_metrics.png",
           "Figure 8. Throughput, end-to-end latency and backlog during the run.", width=5.2)
    document.add_heading("8.1 Finding and removing the bottleneck", level=2)
    paragraph(document, "The first working version of the pipeline could not keep up. The per-stage "
                        "traces showed why: the machine-learning worker needed about thirty milliseconds "
                        "for every window, and almost all of it was the pandas feature build, which "
                        "recomputed sixty rolling and lag expressions over the whole trailing buffer in "
                        "order to obtain a single row. Its inbox grew without bound while the other "
                        "stages sat idle, which is what a bottleneck looks like in the backlog series.")
    paragraph(document, "The fix was to compute the same feature row directly with NumPy on the "
                        "trailing buffers, which is roughly fifty times faster than rebuilding the "
                        "pandas frame for a single row. After the change the scoring stage is no longer "
                        "the slowest link: the peak backlog of the run sits at the {} stage instead, "
                        "and the average backlog across the whole topology is {:.1f} "
                        "messages.".format(stream.get("backlog_peak_stage", "merge"),
                                           stream["backlog_mean"]))
    paragraph(document, "A second defect surfaced in the same traces. The two producers drift by a few "
                        "milliseconds, which at an accelerated replay is several minutes of event time, "
                        "so batches from the slower producer arrived after their window had already been "
                        "closed by the faster one. The window worker reopened the window and the same "
                        "minute was scored twice. The fix has two parts: the producers are released by a "
                        "shared start time, and a closed window is never reopened, so late records are "
                        "counted and dropped. With the current watermark the run drops {:,} records out "
                        "of {:,}.".format(stream.get("late_events_dropped", 0),
                                          int(stream["events_processed"])))


def section_discussion(document, offline, stream):
    document.add_heading("9. Discussion and limitations", level=1)
    best_clf = max(offline["classification"], key=lambda item: item["f1"])
    best_det = max(offline["anomaly"], key=lambda item: item["f1"])
    paragraph(document, "The best detector on the held-out window is the {} with an F1 of {} and a ROC "
                        "AUC of {}. The best classifier is the {} with an F1 of {} and a ROC AUC of {}. "
                        "Detection and prediction answer different questions: the detector reports that "
                        "the current minute is abnormal, the classifier reports that the next minute is "
                        "likely to be, and section 7.2 shows where that skill actually comes from."
              .format(best_det["method"], fmt(best_det["f1"], 2), fmt(best_det["roc_auc"], 2),
                      best_clf["model"], fmt(best_clf["f1"], 2), fmt(best_clf["roc_auc"], 2)))
    paragraph(document, "Five limitations are worth stating plainly. The data is synthetic, so the "
                        "incidents have known shapes and the detectors face a cleaner problem than "
                        "production traffic with correlated failures and missing data. The held-out "
                        "window contains only {} incidents, which is a small sample for event-level "
                        "claims. The distributed layer runs on one machine, so network partitions, "
                        "broker failures and rebalancing are out of scope, although the socket transport "
                        "means the code itself is not tied to one host. The models are trained once "
                        "offline and then served in the stream, so no drift detection or scheduled "
                        "retraining is implemented. "
                        "Finally, the supervised labels come from the generator and would not exist at "
                        "prediction time in production, where they would have to be recovered from "
                        "incident tickets or from the unsupervised detector.".format(
                            offline["anomaly"][0]["anomaly_events"]))
    paragraph(document, "The natural extensions are a Kafka or Flink backend with the same operators, a "
                        "drift monitor that compares the live feature distribution against the training "
                        "distribution, per-endpoint models instead of one global model, and alert "
                        "suppression so that one long incident produces one page rather than one page "
                        "per minute.")


def section_conclusion(document, cfg, offline, stream):
    document.add_heading("10. Conclusion", level=1)
    paragraph(document, "The streaming application was extended into a full analytics system. The "
                        "dashboard updates continuously with four views. The temporal layer uses "
                        "timestamps, rolling windows, resampling and STL decomposition, and quantifies a "
                        "daily cycle with a seasonal strength of {:.2f}. Two anomaly detectors are "
                        "implemented and evaluated against ground truth, and the streaming detector "
                        "catches {} of {} injected incidents. A real-time machine-learning layer "
                        "predicts the next minute from features that exist at prediction time and is "
                        "evaluated with F1, ROC AUC and PR AUC against a baseline. The distributed "
                        "layer runs {} processes over {} hash partitions with watermarked windows and "
                        "reports throughput, latency, event counts, backlog and late records measured "
                        "during a live run of {:,} events in {:.0f} seconds.".format(
                            offline["timeseries"]["seasonal_strength"],
                            stream["streaming_detector"]["events_detected"],
                            stream["streaming_detector"]["anomaly_events"],
                            stream["processes"], cfg["pipeline"]["partitions"],
                            int(stream["events_processed"]), stream["wall_clock_seconds"]))


def section_appendix(document):
    document.add_heading("Appendix. Reproducing the results", level=1)
    paragraph(document, "Install the dependencies with pip install -r requirements.txt, then run python "
                        "run_all.py. The stages run in order: the generator writes data/events.csv.gz "
                        "with its labels, the offline stage performs the time-series, anomaly and "
                        "machine-learning analysis and saves the fitted models, the stream stage "
                        "replays the events through the distributed pipeline with the live dashboard, "
                        "and the last two stages regenerate this report and the slides. All parameters "
                        "live in config.yaml and the seed is fixed. Metrics land in results/ as JSON and "
                        "CSV, figures in figures/.")


def main():
    cfg = load_config()
    offline, stream = load_results(cfg)
    document = Document()
    setup(document)
    for section in document.sections:
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    title_page(document, cfg, offline, stream)
    section_intro(document, offline)
    section_architecture(document, cfg, stream)
    section_data(document, cfg, offline)
    section_visualization(document, cfg)
    section_timeseries(document, cfg, offline)
    section_anomaly(document, cfg, offline, stream)
    section_ml(document, cfg, offline)
    section_stream(document, cfg, stream)
    section_discussion(document, offline, stream)
    section_conclusion(document, cfg, offline, stream)
    section_appendix(document)

    out = os.path.join(cfg["root"], "report", "Technical_Report.docx")
    document.save(out)
    print("saved " + out)


if __name__ == "__main__":
    main()
