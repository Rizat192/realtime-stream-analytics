import multiprocessing as mp
import os
import queue
import threading
import time
import zlib
from multiprocessing.connection import Client, Listener

import joblib
import numpy as np
import pandas as pd

from .anomaly import OnlineRobustDetector
from .features import OnlineFeatureBuilder, merge_partials, partial_aggregate

SENTINEL = "__end__"
AUTH = b"endterm-stream"
HOST = "127.0.0.1"
DEBUG = os.environ.get("STREAM_DEBUG") == "1"


def trace(stage, count, spent):
    if DEBUG and count % 200 == 0:
        print("[{}] {} windows, {:.1f} ms each".format(stage, count, spent / max(count, 1) * 1000),
              flush=True)


def window_port(cfg, partition):
    return cfg["pipeline"]["base_port"] + partition


def merger_port(cfg):
    return cfg["pipeline"]["base_port"] + 10


def scorer_port(cfg, index):
    return cfg["pipeline"]["base_port"] + 20 + index


def sink_port(cfg):
    return cfg["pipeline"]["base_port"] + 30


class Inbox:
    def __init__(self, port, senders):
        self.listener = Listener((HOST, port), authkey=AUTH)
        self.queue = queue.Queue()
        self.senders = senders
        self.acceptor = threading.Thread(target=self.serve, daemon=True)
        self.acceptor.start()

    def serve(self):
        for _ in range(self.senders):
            connection = self.listener.accept()
            threading.Thread(target=self.pump, args=(connection,), daemon=True).start()

    def pump(self, connection):
        while True:
            try:
                message = connection.recv()
            except (EOFError, OSError):
                return
            self.queue.put(message)

    def get(self):
        return self.queue.get()

    def poll(self, timeout):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def depth(self):
        return self.queue.qsize()

    def close(self):
        try:
            self.listener.close()
        except OSError:
            pass


def connect(port, attempts=400):
    for _ in range(attempts):
        try:
            return Client((HOST, port), authkey=AUTH)
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    raise RuntimeError("no listener on port " + str(port))


def partition_of(endpoint, partitions):
    return zlib.crc32(endpoint.encode("utf-8")) % partitions


def load_events(cfg, producer_id):
    path = os.path.join(cfg["paths"]["data"], "events.csv.gz")
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    endpoints = sorted(frame["endpoint"].unique())
    mine = endpoints[producer_id::cfg["pipeline"]["producers"]]
    frame = frame[frame["endpoint"].isin(mine)].copy()
    start = pd.Timestamp(cfg["generator"]["start"])
    frame["minute_idx"] = ((frame["timestamp"] - start).dt.total_seconds() // 60).astype(int)
    return frame.sort_values("minute_idx")


def producer(producer_id, cfg, start_at):
    frame = load_events(cfg, producer_id)
    partitions = cfg["pipeline"]["partitions"]
    tick = cfg["pipeline"]["window_seconds"] / float(cfg["pipeline"]["speed_factor"])
    frame["partition"] = [partition_of(name, partitions) for name in frame["endpoint"]]
    columns = ["endpoint", "latency_ms", "status", "bytes_out", "user_id"]
    outputs = [connect(window_port(cfg, partition)) for partition in range(partitions)]
    delay = start_at - time.time()
    if delay > 0:
        time.sleep(delay)
    next_tick = time.time()
    for minute_idx, group in frame.groupby("minute_idx", sort=True):
        emit_ts = time.time()
        for partition in range(partitions):
            block = group[group["partition"] == partition]
            records = block[columns].to_dict("records") if len(block) else []
            outputs[partition].send({
                "minute_idx": int(minute_idx),
                "producer": producer_id,
                "events": records,
                "produce_ts": emit_ts,
            })
        next_tick += tick
        delay = next_tick - time.time()
        if DEBUG and minute_idx % 400 == 0:
            print("[producer {}] minute {}, lag {:.1f} ms".format(
                producer_id, minute_idx, max(0.0, -delay) * 1000.0), flush=True)
        if delay > 0:
            time.sleep(delay)
    for output in outputs:
        output.send(SENTINEL)
        output.close()


def window_worker(partition, cfg):
    producers = cfg["pipeline"]["producers"]
    watermark_gap = cfg["pipeline"]["watermark_minutes"]
    inbox = Inbox(window_port(cfg, partition), producers)
    output = connect(merger_port(cfg))
    buffers = {}
    stamps = {}
    seen_end = 0
    watermark = -1
    events = 0
    late = 0
    flushed = set()

    def flush(minute_idx):
        flushed.add(minute_idx)
        records = buffers.pop(minute_idx, [])
        produce_ts, arrivals = stamps.pop(minute_idx, (time.time(), 0))
        payload = partial_aggregate(records) if records else empty_partial()
        output.send({
            "minute_idx": minute_idx,
            "partition": partition,
            "partial": payload,
            "produce_ts": produce_ts,
            "window_ts": time.time(),
            "batches": arrivals,
            "depth_window": inbox.depth(),
            "events_seen": events,
            "late_events": late,
        })

    while True:
        message = inbox.get()
        if message == SENTINEL:
            seen_end += 1
            if seen_end == producers:
                for minute_idx in sorted(buffers):
                    flush(minute_idx)
                output.send(SENTINEL)
                output.close()
                inbox.close()
                return
            continue
        minute_idx = message["minute_idx"]
        if minute_idx in flushed:
            late += len(message["events"])
            continue
        buffers.setdefault(minute_idx, []).extend(message["events"])
        events += len(message["events"])
        previous = stamps.get(minute_idx)
        produce_ts = message["produce_ts"] if previous is None else min(previous[0], message["produce_ts"])
        arrivals = 1 if previous is None else previous[1] + 1
        stamps[minute_idx] = (produce_ts, arrivals)
        watermark = max(watermark, minute_idx)
        for ready in sorted(k for k in buffers if k <= watermark - watermark_gap):
            flush(ready)


def empty_partial():
    return {
        "requests": 0,
        "errors": 0,
        "server_errors": 0,
        "latency_sum": 0.0,
        "latency_values": [],
        "bytes_out": 0.0,
        "users": [],
    }


def merger(cfg):
    partitions = cfg["pipeline"]["partitions"]
    watermark_gap = cfg["pipeline"]["watermark_minutes"]
    inbox = Inbox(merger_port(cfg), partitions)
    outputs = [connect(scorer_port(cfg, index)) for index in range(cfg["pipeline"]["scoring_workers"])]
    start = pd.Timestamp(cfg["generator"]["start"])
    buffers = {}
    seen_end = 0
    watermark = -1
    emitted = set()

    def flush(minute_idx):
        entries = buffers.pop(minute_idx, [])
        if not entries or minute_idx in emitted:
            return
        emitted.add(minute_idx)
        row = merge_partials([entry["partial"] for entry in entries])
        message = {
            "minute_idx": minute_idx,
            "minute": str(start + pd.Timedelta(minutes=minute_idx)),
            "row": row,
            "partitions_seen": len(entries),
            "produce_ts": min(entry["produce_ts"] for entry in entries),
            "window_ts": max(entry["window_ts"] for entry in entries),
            "merge_ts": time.time(),
            "depth_window": max(entry["depth_window"] for entry in entries),
            "depth_merge": inbox.depth(),
            "events_seen": sum(entry["events_seen"] for entry in entries),
            "late_events": sum(entry["late_events"] for entry in entries),
        }
        for output in outputs:
            output.send(message)

    while True:
        message = inbox.get()
        if message == SENTINEL:
            seen_end += 1
            if seen_end == partitions:
                for minute_idx in sorted(buffers):
                    flush(minute_idx)
                for output in outputs:
                    output.send(SENTINEL)
                    output.close()
                inbox.close()
                return
            continue
        minute_idx = message["minute_idx"]
        if minute_idx in emitted:
            continue
        buffers.setdefault(minute_idx, []).append(message)
        if len(buffers[minute_idx]) == partitions:
            flush(minute_idx)
            continue
        watermark = max(watermark, minute_idx)
        for ready in sorted(k for k in buffers if k <= watermark - watermark_gap):
            flush(ready)


def anomaly_worker(index, cfg):
    inbox = Inbox(scorer_port(cfg, index), 1)
    output = connect(sink_port(cfg))
    detector = OnlineRobustDetector(cfg)
    bundle = joblib.load(os.path.join(cfg["paths"]["models"], "iforest.joblib"))
    model = bundle["model"]
    history = []
    spent = 0.0
    done = 0
    window = cfg["timeseries"]["rolling_window"]
    while True:
        message = inbox.get()
        if message == SENTINEL:
            output.send(SENTINEL)
            output.close()
            inbox.close()
            return
        began = time.time()
        row = message["row"]
        score, flag = detector.update(row)
        history.append(row)
        if len(history) > window + 1:
            history.pop(0)
        past = history[:-1] if len(history) > 1 else history
        mean_requests = np.mean([item["requests"] for item in past]) or 1.0
        mean_latency = np.mean([item["latency_p95"] for item in past]) or 1.0
        vector = [
            row["requests"], row["latency_p50"], row["latency_p95"], row["error_rate"],
            row["server_error_rate"], row["bytes_per_request"],
            row["requests"] / mean_requests, row["latency_p95"] / mean_latency,
        ]
        sample = np.asarray([vector], dtype=float)
        iforest_score = float(-model.score_samples(sample)[0])
        iforest_flag = int(-iforest_score < model.offset_)
        spent += time.time() - began
        done += 1
        trace("anomaly", done, spent)
        output.send({
            "stage": "anomaly",
            "minute_idx": message["minute_idx"],
            "minute": message["minute"],
            "row": row,
            "robust_score": score,
            "robust_flag": flag,
            "iforest_score": iforest_score,
            "iforest_flag": iforest_flag,
            "produce_ts": message["produce_ts"],
            "window_ts": message["window_ts"],
            "merge_ts": message["merge_ts"],
            "score_ts": time.time(),
            "depth_window": message["depth_window"],
            "depth_merge": message["depth_merge"],
            "depth_score": inbox.depth(),
            "events_seen": message["events_seen"],
            "late_events": message["late_events"],
        })


def ml_worker(index, cfg):
    inbox = Inbox(scorer_port(cfg, index), 1)
    output = connect(sink_port(cfg))
    bundle = joblib.load(os.path.join(cfg["paths"]["models"], "stream_models.joblib"))
    builder = OnlineFeatureBuilder(cfg)
    columns = bundle["columns"]
    spent = 0.0
    done = 0
    while True:
        message = inbox.get()
        if message == SENTINEL:
            output.send(SENTINEL)
            output.close()
            inbox.close()
            return
        began = time.time()
        builder.push(message["minute"], message["row"])
        proba = 0.0
        if builder.ready():
            features = builder.latest_vector(columns)
            proba = float(bundle["classifier"].predict_proba(features)[0, 1])
        spent += time.time() - began
        done += 1
        trace("ml", done, spent)
        output.send({
            "stage": "ml",
            "minute_idx": message["minute_idx"],
            "minute": message["minute"],
            "incident_proba": proba,
            "threshold": bundle["threshold"],
            "produce_ts": message["produce_ts"],
            "window_ts": message["window_ts"],
            "merge_ts": message["merge_ts"],
            "score_ts": time.time(),
            "depth_score_ml": inbox.depth(),
        })


class StreamRuntime:
    def __init__(self, cfg):
        self.cfg = cfg
        self.context = mp.get_context("spawn")
        self.inbox = None
        self.workers = []

    def start(self):
        cfg = self.cfg
        ctx = self.context
        self.inbox = Inbox(sink_port(cfg), cfg["pipeline"]["scoring_workers"])
        self.workers.append(ctx.Process(target=anomaly_worker, args=(0, cfg), name="scorer-anomaly"))
        self.workers.append(ctx.Process(target=ml_worker, args=(1, cfg), name="scorer-ml"))
        self.workers.append(ctx.Process(target=merger, args=(cfg,), name="merger"))
        for partition in range(cfg["pipeline"]["partitions"]):
            self.workers.append(ctx.Process(target=window_worker, args=(partition, cfg),
                                            name="window-" + str(partition)))
        start_at = time.time() + cfg["pipeline"]["startup_grace"]
        for producer_id in range(cfg["pipeline"]["producers"]):
            self.workers.append(ctx.Process(target=producer, args=(producer_id, cfg, start_at),
                                            name="producer-" + str(producer_id)))
        for worker in self.workers:
            worker.start()

    def drain(self, timeout=0.05, limit=4000):
        batch = []
        first = self.inbox.poll(timeout)
        if first is None:
            return batch
        batch.append(first)
        while len(batch) < limit:
            item = self.inbox.poll(0.0)
            if item is None:
                break
            batch.append(item)
        return batch

    def sink_depth(self):
        return self.inbox.depth()

    def failed(self):
        return [worker.name for worker in self.workers
                if worker.exitcode is not None and worker.exitcode != 0]

    def stop(self):
        for worker in self.workers:
            worker.join(timeout=15)
            if worker.is_alive():
                worker.terminate()
        if self.inbox is not None:
            self.inbox.close()
