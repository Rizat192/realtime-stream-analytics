import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for key, value in cfg["paths"].items():
        absolute = os.path.join(ROOT, value)
        os.makedirs(absolute, exist_ok=True)
        cfg["paths"][key] = absolute
    cfg["root"] = ROOT
    return cfg


def path_in(cfg, kind, name):
    return os.path.join(cfg["paths"][kind], name)
