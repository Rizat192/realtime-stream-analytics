import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, path_in
from src.features import aggregate_events
from src.generator import generate


def main():
    cfg = load_config()
    events, truth = generate(cfg)
    minutes = aggregate_events(events, cfg["timeseries"]["freq"])

    events_path = path_in(cfg, "data", "events.csv.gz")
    truth_path = path_in(cfg, "data", "labels.csv")
    minutes_path = path_in(cfg, "data", "minutes.csv")

    events.to_csv(events_path, index=False, compression="gzip")
    truth.to_csv(truth_path, index=False)
    minutes.to_csv(minutes_path)

    print(f"events            : {len(events):,}")
    print(f"minute windows    : {len(minutes):,}")
    print(f"anomalous minutes : {int(truth['is_anomaly'].sum()):,}")
    print(f"span              : {events['timestamp'].min()} .. {events['timestamp'].max()}")
    print(f"written           : {events_path}")


if __name__ == "__main__":
    main()
