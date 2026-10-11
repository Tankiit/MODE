"""Load completed runs (directories with a DONE marker) for the analyses."""
from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_runs(root: str) -> list[dict]:
    """
    One record per completed configuration. If the same configuration (config
    hash) was completed several times, the most recent directory is used.
    """
    latest: dict[str, dict] = {}
    for done in sorted(Path(root).glob("*/DONE")):
        d = done.parent
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        hist = json.loads((d / "history.json").read_text())
        name = cfg.get("name") or d.name
        sel = cfg["selection"]
        rec = {
            "dir": d, "config": cfg, "history": hist,
            "group": name.split("_")[0],
            "model": cfg["model"]["name"], "data": cfg["data"]["name"],
            "method": sel["method"], "alpha": sel["alpha"],
            "steps": cfg["training"]["steps"], "seed": cfg["seed"],
            "final_val_ppl": hist["final_val_ppl"],
            "wall_time_h": hist["wall_time_s"] / 3600.0,
            "mtime": done.stat().st_mtime,
        }
        h = cfg["config_hash"]
        if h not in latest or rec["mtime"] > latest[h]["mtime"]:
            latest[h] = rec
    return list(latest.values())


def select(records: list[dict], **key) -> list[dict]:
    return [r for r in records if all(r[k] == v for k, v in key.items())]
