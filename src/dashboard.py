import matplotlib

import numpy as np


def pick_backend():
    for name in ["TkAgg", "QtAgg"]:
        try:
            matplotlib.use(name, force=True)
            return True
        except Exception:
            continue
    return False


class LiveDashboard:
    def __init__(self, cfg, interactive=True, span=240):
        self.cfg = cfg
        self.span = span
        self.interactive = interactive and pick_backend()
        if not self.interactive:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25})
        self.figure, self.axes = plt.subplots(2, 2, figsize=(13, 7))
        self.figure.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.08,
                                    hspace=0.30, wspace=0.22)
        self.figure.canvas.manager.set_window_title("Real time API stream monitor")
        self.ops_twin = self.axes[1][1].twinx()
        self.proba_twin = self.axes[1][0].twinx()
        if interactive:
            plt.ion()
            plt.show(block=False)

    def update(self, state, header):
        minutes = state["minute"][-self.span:]
        if not minutes:
            return
        requests = state["requests"][-self.span:]
        rolling = state["requests_roll"][-self.span:]
        latency = state["latency_p95"][-self.span:]
        errors = [value * 100 for value in state["error_rate"][-self.span:]]
        proba = state["incident_proba"][-self.span:]
        flags = np.asarray(state["anomaly_flag"][-self.span:], dtype=bool)
        x = np.arange(len(minutes))

        ax = self.axes[0][0]
        ax.clear()
        ax.plot(x, requests, color="#7f7f7f", lw=0.8, label="requests per minute")
        ax.plot(x, rolling, color="#1f77b4", lw=1.4, label="15 minute rolling mean")
        ax.set_title("stream volume")
        ax.legend(fontsize=6.5, loc="upper left")

        ax = self.axes[0][1]
        ax.clear()
        ax.plot(x, latency, color="#7f7f7f", lw=0.9, label="p95 latency (ms)")
        if flags.any():
            ax.scatter(x[flags], np.asarray(latency)[flags], s=14, color="#c0392b",
                       zorder=3, label="anomaly alarm")
        ax.set_title("tail latency and alarms")
        ax.legend(fontsize=6.5, loc="upper left")

        ax = self.axes[1][0]
        ax.clear()
        self.proba_twin.clear()
        ax.plot(x, errors, color="#2e8b57", lw=1.0, label="error rate (%)")
        self.proba_twin.plot(x, proba, color="#1f77b4", lw=1.0, label="incident probability")
        self.proba_twin.axhline(state["threshold"], color="#c0392b", ls="--", lw=0.9)
        self.proba_twin.set_ylim(0, 1.02)
        ax.set_title("errors and predicted incident risk")
        ax.legend(fontsize=6.5, loc="upper left")
        self.proba_twin.legend(fontsize=6.5, loc="upper right")

        ax = self.axes[1][1]
        ax.clear()
        self.ops_twin.clear()
        elapsed = state["ops_elapsed"][-self.span:]
        throughput = state["ops_throughput"][-self.span:]
        backlog = state["ops_backlog"][-self.span:]
        if elapsed:
            ax.plot(elapsed, throughput, color="#1f77b4", lw=1.0, label="events per second")
            self.ops_twin.plot(elapsed, backlog, color="#c0392b", lw=1.0, label="backlog")
            ax.set_xlabel("elapsed seconds")
            ax.legend(fontsize=6.5, loc="upper left")
            self.ops_twin.legend(fontsize=6.5, loc="upper right")
        ax.set_title("operational metrics")

        self.figure.suptitle(header, fontsize=9)
        if self.interactive:
            self.figure.canvas.draw_idle()
            self.plt.pause(0.001)

    def save(self, path):
        self.figure.savefig(path, dpi=130, bbox_inches="tight")

    def close(self):
        self.plt.close(self.figure)
