#!/usr/bin/env python3
"""
Expand a sweep file (e.g. configs/sweeps/paper.yaml) into one fully resolved
config file per run, plus run lists for Slurm arrays:

  <out>/<run name>.yaml     self-contained config (no include)
  <out>/list.txt            all config paths, in sweep order
  <out>/list_<model>.txt    the same, per model (separate arrays / time limits)

Run names are <group>_<data>_<model>_<method>_a<alpha>_st<steps>_s<seed>.
Every config is validated; duplicate names or duplicate configurations are
errors. Usage:
  python scripts/expand_sweep.py configs/sweeps/paper.yaml --out sweeps/paper
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sieve.config import (RunConfig, _build, _read_yaml, config_hash, deep_merge,  # noqa: E402
                          to_dict, validate)

INCLUDE_DIRS = {"model": "models", "data": "data", "method": "methods"}


def nest(key: str, value) -> dict:
    node = value
    for part in reversed(key.split(".")):
        node = {part: node}
    return node


def matches(when: dict, labels: dict) -> bool:
    for k, cond in when.items():
        v = labels.get(k)
        if isinstance(cond, list) and v not in cond:
            return False
        if not isinstance(cond, list) and v != cond:
            return False
    return True


def expand(sweep_path: Path) -> list[tuple[str, dict]]:
    sweep = yaml.safe_load(sweep_path.read_text())
    configs_root = sweep_path.resolve().parent.parent
    base: dict = {}
    for f in sweep.get("base", []):
        base = deep_merge(base, _read_yaml(sweep_path.parent / f))
    runs = []
    for g in sweep["groups"]:
        inc, grid = g.get("include", {}), g.get("grid", {})
        unknown = set(inc) - set(INCLUDE_DIRS)
        if unknown:
            raise ValueError(f"group {g['name']}: unknown include keys {sorted(unknown)}")
        inc_keys, grid_keys = list(inc), list(grid)
        for inc_vals in itertools.product(*(inc[k] for k in inc_keys)):
            for grid_vals in itertools.product(*(grid[k] for k in grid_keys)):
                for z in g.get("zip", [{}]):
                    d = base
                    for k, v in zip(inc_keys, inc_vals):
                        d = deep_merge(d, _read_yaml(configs_root / INCLUDE_DIRS[k] / f"{v}.yaml"))
                    overrides = {**g.get("set", {}), **dict(zip(grid_keys, grid_vals)), **z}
                    for k, v in overrides.items():
                        d = deep_merge(d, nest(k, v))
                    labels = {**dict(zip(inc_keys, inc_vals)), **overrides}
                    for rule in g.get("rules", []):
                        if matches(rule["when"], labels):
                            for k, v in rule["set"].items():
                                d = deep_merge(d, nest(k, v))
                    d = {**d, "name": None}
                    cfg = _build(RunConfig, d, "")
                    d["name"] = f"{g['name']}_{cfg.run_name}"
                    runs.append((d["name"], d))
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true", help="allow writing into an existing directory")
    args = p.parse_args()

    runs = expand(Path(args.sweep))
    names = [n for n, _ in runs]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"duplicate run names: {dup[:5]}")
    hashes = {}
    for n, d in runs:
        cfg = _build(RunConfig, d, "")
        validate(cfg)
        h = config_hash(cfg)
        if h in hashes:
            raise SystemExit(f"{n} and {hashes[h]} have the same configuration")
        hashes[h] = n

    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty (use --force to rewrite it)")
    out.mkdir(parents=True, exist_ok=True)
    per_model = defaultdict(list)
    paths = []
    for n, d in runs:
        path = out / f"{n}.yaml"
        path.write_text(yaml.safe_dump(to_dict(_build(RunConfig, d, "")), sort_keys=False))
        paths.append(str(path))
        per_model[d["model"]["name"]].append(str(path))
    (out / "list.txt").write_text("\n".join(paths) + "\n")
    for model, ps in per_model.items():
        (out / f"list_{model}.txt").write_text("\n".join(ps) + "\n")
    print(f"{len(runs)} runs → {out}/")
    for model, ps in per_model.items():
        print(f"  list_{model}.txt: {len(ps)} runs")


if __name__ == "__main__":
    main()
