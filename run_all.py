import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    ("dataset", ["scripts/make_dataset.py"]),
    ("offline analysis", ["scripts/run_offline.py"]),
    ("stream demonstration", ["scripts/run_stream.py", "--headless"]),
    ("technical report", ["scripts/make_report.py"]),
    ("presentation", ["scripts/make_slides.py"]),
]


def main():
    for name, command in STAGES:
        print("=" * 70)
        print("stage: " + name)
        print("=" * 70)
        result = subprocess.run([sys.executable] + command, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit("stage failed: " + name)
    print("all stages complete")


if __name__ == "__main__":
    main()
