import os

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_name, "1")

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import plots
from src.anomaly import event_recall, evaluate
from src.config import load_config, path_in
from src.dashboard import LiveDashboard
from src.pipeline import SENTINEL, StreamRuntime

STAGE_KEYS = ["window", "merge", "anomaly", "ml", "sink"]


def new_state():
    return {
        "minute": [], "requests": [], "requests_roll": [],
        "latency_p95": [], "error_rate": [], "incident_proba": [], "anomaly_flag": [],
        "ops_elapsed": [], "ops_throughput": [], "ops_backlog": [], "threshold": 0.5,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="run the distributed stream demonstration")
    parser.add_argument("--headless", action="store_true", help="render without an interactive window")
    parser.add_argument("--speed", type=float, default=None, help="replay speed factor")
    parser.add_argument("--span", type=int, default=240, help="minutes visible in the dashboard")
    parser.add_argument("--draw-every", type=float, default=None, help="seconds between redraws")
    return parser.parse_args()


def merge_stage_results(anomaly_message, ml_message, now):
    row = anomaly_message["row"]
    produce_ts = min(anomaly_message["produce_ts"], ml_message["produce_ts"])
    score_ts = max(anomaly_message["score_ts"], ml_message["score_ts"])
    return {
        "minute_idx": anomaly_message["minute_idx"],
        "minute": anomaly_message["minute"],
        "requests": row["requests"],
        "error_rate": row["error_rate"],
        "server_error_rate": row["server_error_rate"],
        "latency_p50": row["latency_p50"],
        "latency_p95": row["latency_p95"],
        "unique_users": row["unique_users"],
        "robust_score": anomaly_message["robust_score"],
        "robust_flag": anomaly_message["robust_flag"],
        "iforest_score": anomaly_message["iforest_score"],
        "iforest_flag": anomaly_message["iforest_flag"],
        "incident_proba": ml_message["incident_proba"],
        "threshold": ml_message["threshold"],
        "events_seen": anomaly_message["events_seen"],
        "late_events": anomaly_message["late_events"],
        "depth_window": anomaly_message["depth_window"],
        "depth_merge": anomaly_message["depth_merge"],
        "depth_anomaly": anomaly_message["depth_score"],
        "depth_ml": ml_message["depth_score_ml"],
        "window_ms": (anomaly_message["window_ts"] - produce_ts) * 1000.0,
        "merge_ms": (anomaly_message["merge_ts"] - anomaly_message["window_ts"]) * 1000.0,
        "score_ms": (score_ts - anomaly_message["merge_ts"]) * 1000.0,
        "end_to_end_ms": (now - produce_ts) * 1000.0,
    }


def push_state(state, merged, cfg):
    state["minute"].append(merged["minute"])
    state["requests"].append(merged["requests"])
    window = cfg["timeseries"]["rolling_window"]
    state["requests_roll"].append(float(np.mean(state["requests"][-window:])))
    state["latency_p95"].append(merged["latency_p95"])
    state["error_rate"].append(merged["error_rate"])
    state["incident_proba"].append(merged["incident_proba"])
    state["anomaly_flag"].append(merged["robust_flag"])
    state["threshold"] = merged["threshold"]


def main():
    args = parse_args()
    cfg = load_config()
    if args.speed:
        cfg["pipeline"]["speed_factor"] = args.speed
    for name in ["iforest.joblib", "stream_models.joblib"]:
        if not os.path.exists(path_in(cfg, "models", name)):
            raise SystemExit("missing model " + name + ", run scripts/run_offline.py first")

    draw_every = args.draw_every or (5.0 if args.headless else 0.5)
    debug = os.environ.get("STREAM_DEBUG") == "1"
    runtime = StreamRuntime(cfg)
    dashboard = LiveDashboard(cfg, interactive=not args.headless, span=args.span)
    state = new_state()

    pending = {}
    records = []
    ops_rows = []
    latencies = []
    depths = {key: 0 for key in STAGE_KEYS}
    events_total = 0
    started = time.time()
    last_metrics = started
    last_draw = started
    last_events = 0
    finished = 0
    runtime.start()

    while finished < cfg["pipeline"]["scoring_workers"]:
        batch = runtime.drain()
        now = time.time()
        for message in batch:
            if message == SENTINEL:
                finished += 1
                continue
            key = message["minute_idx"]
            slot = pending.setdefault(key, {})
            slot[message["stage"]] = message
            if "anomaly" in slot and "ml" in slot:
                merged = merge_stage_results(slot.pop("anomaly"), slot.pop("ml"), now)
                pending.pop(key, None)
                records.append(merged)
                latencies.append(merged["end_to_end_ms"])
                events_total = max(events_total, int(merged["events_seen"]))
                depths["window"] = merged["depth_window"]
                depths["merge"] = merged["depth_merge"]
                depths["anomaly"] = merged["depth_anomaly"]
                depths["ml"] = merged["depth_ml"]
                push_state(state, merged, cfg)

        if now - last_metrics >= cfg["pipeline"]["metrics_interval"]:
            depths["sink"] = runtime.sink_depth()
            backlog_total = sum(depths.values())
            span = now - last_metrics
            throughput = (events_total - last_events) / span if span > 0 else 0.0
            recent = latencies[-200:] if latencies else [0.0]
            row = {
                "elapsed_s": round(now - started, 2),
                "events_processed": events_total,
                "throughput_eps": round(throughput, 1),
                "minutes_processed": len(records),
                "backlog_total": backlog_total,
                "latency_ms_p50": round(float(np.percentile(recent, 50)), 2),
                "latency_ms_p95": round(float(np.percentile(recent, 95)), 2),
            }
            row.update({"backlog_" + key: depths[key] for key in STAGE_KEYS})
            ops_rows.append(row)
            state["ops_elapsed"].append(row["elapsed_s"])
            state["ops_throughput"].append(row["throughput_eps"])
            state["ops_backlog"].append(backlog_total)
            last_events = events_total
            last_metrics = now
            broken = runtime.failed()
            if broken:
                raise SystemExit("stage failed: " + ", ".join(broken))

        if now - last_draw >= draw_every:
            header = ("minutes {m} | events {e:,} | throughput {t:,.0f} eps | "
                      "end to end p95 {l:.0f} ms | backlog {b} | elapsed {s:.0f} s").format(
                m=len(records), e=events_total,
                t=state["ops_throughput"][-1] if state["ops_throughput"] else 0.0,
                l=float(np.percentile(latencies[-200:], 95)) if latencies else 0.0,
                b=state["ops_backlog"][-1] if state["ops_backlog"] else 0,
                s=now - started)
            dashboard.update(state, header)
            last_draw = now
            if debug:
                print("[sink] " + header + " | depths " + str(depths)
                      + " | pending " + str(len(pending)), flush=True)

    runtime.stop()
    elapsed = time.time() - started
    frame = pd.DataFrame(records).sort_values("minute_idx").reset_index(drop=True)
    ops = pd.DataFrame(ops_rows)
    frame.to_csv(path_in(cfg, "results", "stream_results.csv"), index=False)
    ops.to_csv(path_in(cfg, "results", "ops_metrics.csv"), index=False)

    dashboard.update(state, "final stream state after {:.0f} seconds".format(elapsed))
    dashboard.save(path_in(cfg, "figures", "fig_dashboard.png"))
    dashboard.close()
    plots.ops_figure(ops, path_in(cfg, "figures", "fig_ops_metrics.png"))

    summary = summarize(cfg, frame, ops, elapsed)
    with open(path_in(cfg, "results", "stream_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def summarize(cfg, frame, ops, elapsed):
    truth = pd.read_csv(path_in(cfg, "data", "labels.csv"), parse_dates=["minute"])
    truth["minute_idx"] = range(len(truth))
    joined = frame.merge(truth[["minute_idx", "is_anomaly"]], on="minute_idx", how="left")
    joined["is_anomaly"] = joined["is_anomaly"].fillna(0).astype(int)
    warm = joined.iloc[cfg["anomaly"]["zscore_min_periods"]:]
    detector = evaluate(warm["is_anomaly"].values, warm["robust_score"].values,
                        warm["robust_flag"].values, "streaming robust z score")
    detector.update(event_recall(warm, warm["robust_flag"].values))
    return {
        "wall_clock_seconds": round(elapsed, 2),
        "speed_factor": cfg["pipeline"]["speed_factor"],
        "simulated_minutes": int(len(frame)),
        "simulated_hours": round(len(frame) / 60.0, 2),
        "events_processed": int(ops["events_processed"].max()) if len(ops) else 0,
        "mean_throughput_eps": round(float(ops["throughput_eps"].mean()), 1) if len(ops) else 0.0,
        "peak_throughput_eps": round(float(ops["throughput_eps"].max()), 1) if len(ops) else 0.0,
        "windows_per_second": round(len(frame) / elapsed, 2),
        "end_to_end_ms_mean": round(float(frame["end_to_end_ms"].mean()), 2),
        "end_to_end_ms_p95": round(float(frame["end_to_end_ms"].quantile(0.95)), 2),
        "window_stage_ms_mean": round(float(frame["window_ms"].mean()), 2),
        "merge_stage_ms_mean": round(float(frame["merge_ms"].mean()), 2),
        "score_stage_ms_mean": round(float(frame["score_ms"].mean()), 2),
        "backlog_mean": round(float(ops["backlog_total"].mean()), 1) if len(ops) else 0.0,
        "backlog_peak": int(ops["backlog_total"].max()) if len(ops) else 0,
        "backlog_peak_stage": max(STAGE_KEYS, key=lambda key: int(ops["backlog_" + key].max()))
        if len(ops) else "none",
        "late_events_dropped": int(frame["late_events"].max()),
        "watermark_minutes": cfg["pipeline"]["watermark_minutes"],
        "processes": (cfg["pipeline"]["producers"] + cfg["pipeline"]["partitions"]
                      + 1 + cfg["pipeline"]["scoring_workers"]),
        "streaming_detector": detector,
    }


if __name__ == "__main__":
    main()
