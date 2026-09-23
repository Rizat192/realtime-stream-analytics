# Real-Time Stream Analytics

A monitoring system for API traffic. It builds a realistic stream of server requests, runs that
stream through a distributed pipeline, finds incidents while they are happening, and shows them
on a live dashboard.

## The data

Every record is one HTTP request: a timestamp, the endpoint, the region, the latency, the status
code, the response size and a user id. Six endpoints produce about 271,000 events across three
simulated days, which are grouped into 4,320 one-minute windows.

Thirty incidents are planted in that stream on purpose, and their true start and end times are
recorded. They come in four kinds: latency spikes, bursts of errors, sudden drops in traffic and
sudden surges. Because the answers are known in advance, every detector can be scored against
the truth instead of judged by eye.

## How a window is built

Each one-minute window is summarised into features: request counts, latency percentiles, error
rates and so on. The features for a window are built only from data up to that window, never
from later data. This matters. If a feature quietly used information from the future, the model
would look excellent on paper and fail completely in production.

## Finding incidents

Two detectors run side by side. The first is a rolling robust z-score, which compares the current
window against the recent past using the median and the median absolute deviation, so a single
extreme value does not poison the baseline. The second is an isolation forest, which learns what
a normal window looks like across all features at once.

On top of those, a gradient boosting classifier looks one minute ahead: given the features of the
current window, it predicts whether the next minute will be part of an incident. It is scored
separately on the minutes where an incident starts and on the minutes where one is already
running, because catching a problem as it begins is the part that actually matters. Permutation
importance shows which features the prediction really depends on.

## The pipeline

Nine separate operating system processes, connected only by authenticated TCP connections on the
loopback interface. No process shares memory with any other, so any stage could be moved to a
different machine by changing one host name.

Two producers replay the event stream and split it into four partitions using a stable CRC32 hash
of the endpoint, so the same endpoint always lands in the same partition. Four window workers
aggregate one-minute tumbling windows inside their own partition, holding results back behind a
watermark of twenty event-time minutes so that late events still get counted. A merger joins the
four partial results into one complete window and sends it to two scoring workers: one runs the
anomaly detectors, the other runs the classifier. A sink in the main process joins both result
streams, drives the dashboard, and records throughput, end-to-end latency, how many events were
processed, the backlog at each stage and how many late events had to be dropped.

## Running it

```
pip install -r requirements.txt
python run_all.py
```

Or one stage at a time:

```
python scripts/make_dataset.py
python scripts/run_offline.py
python scripts/run_stream.py
```

`scripts/run_stream.py --headless` runs without an interactive window and still saves the
dashboard image. `--speed 7200` replays the stream faster and `--span 400` widens the visible
window.

Everything is seeded from `config.yaml` (`seed: 42`), so running it again produces the same
dataset, the same models and the same numbers.

## Files

```
config.yaml              every parameter in one place
src/generator.py         the traffic generator with labelled incidents
src/features.py          window aggregation and feature building
src/timeseries.py        rolling windows, resampling, STL decomposition, stationarity tests
src/anomaly.py           rolling robust z-score and isolation forest
src/ml.py                the incident classifier and its evaluation
src/pipeline.py          producers, partitions, window workers, merger, scorers, sink
src/dashboard.py         the live four-panel dashboard
src/plots.py             the report figures
scripts/make_dataset.py  writes data/events.csv.gz, data/minutes.csv, data/labels.csv
scripts/run_offline.py   runs the offline analysis and saves the trained models
scripts/run_stream.py    runs the pipeline with the live dashboard
scripts/make_report.py   writes report/Technical_Report.docx from the saved results
scripts/make_slides.py   writes presentation/Endterm_Presentation.pptx
run_all.py               runs every stage in order
```

Results land in `data/` (the dataset and labels), `models/` (the fitted detectors), `results/`
(metrics as JSON and CSV), `figures/` (all report figures), `report/` and `presentation/`.

## Note

This started as a university endterm project. The code, the pipeline design and the written
report are my own work.
