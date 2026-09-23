import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve, roc_curve

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#1f77b4"
ORANGE = "#e07b39"
RED = "#c0392b"
GREEN = "#2e8b57"
GREY = "#7f7f7f"


def shade_truth(ax, index, labels, label="true anomaly"):
    labels = np.asarray(labels, dtype=int)
    start = None
    shown = False
    for position, value in enumerate(labels):
        if value == 1 and start is None:
            start = position
        if value == 0 and start is not None:
            ax.axvspan(index[start], index[position], color=RED, alpha=0.13,
                       label=None if shown else label)
            shown = True
            start = None
    if start is not None:
        ax.axvspan(index[start], index[-1], color=RED, alpha=0.13,
                   label=None if shown else label)


def timeseries_overview(view, truth, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    index = view.index
    axes[0].plot(index, view["requests"], color=GREY, lw=0.5, alpha=0.6, label="requests/min")
    axes[0].plot(index, view["requests_roll_mean"], color=BLUE, lw=1.4, label="15 min rolling mean")
    axes[0].plot(index, view["requests_trend"], color=ORANGE, lw=1.6, label="60 min trend")
    axes[0].set_ylabel("requests / min")
    axes[1].plot(index, view["latency_p95"], color=GREY, lw=0.5, alpha=0.6, label="p95 latency")
    axes[1].plot(index, view["latency_roll_mean"], color=BLUE, lw=1.4, label="15 min rolling mean")
    axes[1].set_ylabel("p95 latency (ms)")
    axes[2].plot(index, view["error_rate"] * 100, color=GREY, lw=0.5, alpha=0.6, label="error rate")
    axes[2].plot(index, view["error_roll_mean"] * 100, color=BLUE, lw=1.4, label="15 min rolling mean")
    axes[2].set_ylabel("error rate (%)")
    for ax in axes:
        shade_truth(ax, index, truth["is_anomaly"].values)
        ax.legend(loc="upper left", fontsize=7, ncol=4)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    fig.suptitle("API traffic stream: level, latency and errors per minute")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def stl_figure(parts, path):
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, name, color in zip(axes, ["observed", "trend", "seasonal", "resid"],
                               [GREY, ORANGE, BLUE, GREEN]):
        ax.plot(parts.index, parts[name], color=color, lw=0.8)
        ax.set_ylabel(name)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    fig.suptitle("STL decomposition of requests per minute, daily period of 1440 minutes")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def seasonal_profiles(by_hour, by_dow, hourly, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    axes[0].bar(by_hour.index, by_hour.values, color=BLUE)
    axes[0].set_title("mean requests per minute by hour")
    axes[0].set_xlabel("hour of day")
    axes[1].bar(by_dow.index, by_dow.values, color=ORANGE)
    axes[1].set_title("mean requests per minute by weekday")
    axes[1].set_xlabel("0 = Monday")
    axes[2].plot(hourly.index, hourly["requests"], color=GREEN, lw=1.2)
    axes[2].set_title("hourly resampled request volume")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def anomaly_figure(minutes, truth, results, threshold, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    index = minutes.index
    axes[0].plot(index, minutes["requests"], color=GREY, lw=0.6)
    axes[0].set_ylabel("requests / min")
    axes[1].plot(index, minutes["latency_p95"], color=GREY, lw=0.6)
    axes[1].set_ylabel("p95 latency (ms)")
    axes[2].plot(index, results["robust_score"], color=BLUE, lw=0.7, label="robust z score")
    axes[2].axhline(threshold, color=RED, ls="--", lw=1.0, label="threshold")
    axes[2].set_ylabel("anomaly score")
    axes[2].set_ylim(0, min(40.0, float(np.nanmax(results["robust_score"])) * 1.05 + 1))

    flags = results["robust_flag"].astype(bool).values
    axes[0].scatter(index[flags], minutes["requests"].values[flags], s=10, color=RED,
                    zorder=3, label="robust z alarm")
    iflags = results["iforest_flag"].astype(bool).values
    axes[1].scatter(index[iflags], minutes["latency_p95"].values[iflags], s=10, color=GREEN,
                    zorder=3, label="isolation forest alarm")
    for ax in axes:
        shade_truth(ax, index, truth["is_anomaly"].values)
        ax.legend(loc="upper left", fontsize=7, ncol=3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    fig.suptitle("Anomaly detection on the minute aggregated stream")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def curves_figure(pairs, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for name, y_true, y_score, color in pairs:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axes[0].plot(fpr, tpr, color=color, lw=1.3, label=name)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        axes[1].plot(recall, precision, color=color, lw=1.3, label=name)
    axes[0].plot([0, 1], [0, 1], color=GREY, ls="--", lw=0.8)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("precision recall")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def classification_timeline(index, proba, y_true, threshold, path):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(index, proba, color=BLUE, lw=0.8, label="predicted incident probability")
    ax.axhline(threshold, color=RED, ls="--", lw=1.0, label="decision threshold")
    shade_truth(ax, index, y_true, label="true incident at t+1")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=7, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_title("Real-time incident classifier on the held out test window")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def importance_figure(frame, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(frame["feature"][::-1], frame["roc_auc_drop"][::-1], color=BLUE)
    ax.set_xlabel("ROC AUC drop when the feature is permuted")
    ax.set_title("Classifier feature importance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ops_figure(frame, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    axes[0].plot(frame["elapsed_s"], frame["throughput_eps"], color=BLUE, lw=1.0)
    axes[0].set_ylabel("events / s")
    axes[0].set_title("throughput")
    axes[1].plot(frame["elapsed_s"], frame["latency_ms_p50"], color=GREEN, lw=1.0, label="p50")
    axes[1].plot(frame["elapsed_s"], frame["latency_ms_p95"], color=ORANGE, lw=1.0, label="p95")
    axes[1].set_ylabel("ms")
    axes[1].set_title("end to end processing latency")
    axes[1].legend(fontsize=7)
    axes[2].plot(frame["elapsed_s"], frame["backlog_total"], color=RED, lw=1.0)
    axes[2].set_ylabel("queued events")
    axes[2].set_title("backlog across partitions")
    axes[2].set_xlabel("elapsed seconds of the stream run")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def architecture_figure(cfg, path):
    partitions = cfg["pipeline"]["partitions"]
    scorers = cfg["pipeline"]["scoring_workers"]
    producers = cfg["pipeline"]["producers"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 46)
    ax.axis("off")
    stages = [
        (2, "event source", ["dataset\nreplay"], BLUE),
        (20, str(producers) + " producers", ["producer " + str(i) for i in range(producers)], ORANGE),
        (38, str(partitions) + " partitions", ["queue p" + str(i) for i in range(partitions)], GREY),
        (56, str(partitions) + " window workers", ["window w" + str(i) for i in range(partitions)], GREEN),
        (74, "merger", ["minute\nmerge"], BLUE),
        (92, str(scorers) + " scoring workers", ["scorer " + str(i) for i in range(scorers)], RED),
    ]
    centers = {}
    for x, title, boxes, color in stages:
        ax.text(x + 5, 42, title, ha="center", fontsize=8.5, weight="bold")
        height = 36.0 / max(len(boxes), 1)
        positions = []
        for i, name in enumerate(boxes):
            y = 38 - (i + 1) * height
            ax.add_patch(FancyBboxPatch((x, y), 10, height * 0.66,
                                        boxstyle="round,pad=0.3", linewidth=1.1,
                                        edgecolor=color, facecolor=color, alpha=0.20))
            ax.text(x + 5, y + height * 0.33, name, ha="center", va="center", fontsize=7.0)
            positions.append((x, x + 10, y + height * 0.33))
        centers[title] = positions
    keys = [item[1] for item in stages]
    for left, right in zip(keys, keys[1:]):
        for _, x_end, y_start in centers[left]:
            for x_begin, _, y_end in centers[right]:
                ax.add_patch(FancyArrowPatch((x_end, y_start), (x_begin, y_end),
                                             arrowstyle="->", mutation_scale=6,
                                             color=GREY, alpha=0.45, lw=0.6))
    ax.text(52, 1.5, "scored minute windows return on the results queue to the dashboard and the metrics sink",
            ha="center", fontsize=8, style="italic")
    ax.set_title("Distributed stream processing topology")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
