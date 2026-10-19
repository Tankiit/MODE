from __future__ import annotations

import modal


app = modal.App("sieve-prepare-bigram-freq")
volume_data = modal.Volume.from_name("sieve-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "torch")


@app.function(image=image, timeout=45 * 60, volumes={"/data": volume_data})
def prepare(dataset: str = "wikitext103", vocab_size: int = 50257) -> dict:
    from pathlib import Path

    import numpy as np
    import torch

    dataset_dir = "openwebmath_1b" if dataset == "owm" else dataset
    data_dir = Path("/data") / dataset_dir
    out_path = data_dir / "bigram_freq.pt"
    if out_path.exists():
        obj = torch.load(out_path, map_location="cpu", weights_only=False)
        return {
            "status": "exists",
            "path": str(out_path),
            "n_bigrams": int(obj["keys"].numel()),
        }

    arr = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    prev = np.asarray(arr[:-1], dtype=np.int64)
    target = np.asarray(arr[1:], dtype=np.int64)
    keys = prev * int(vocab_size) + target
    uniq, counts = np.unique(keys, return_counts=True)
    total = float(counts.sum())
    log_freq = np.log(counts.astype(np.float64) / total).astype(np.float32)

    obj = {
        "keys": torch.from_numpy(uniq.astype(np.int64)),
        "log_freq": torch.from_numpy(log_freq),
        "vocab_size": int(vocab_size),
        "fallback_log_freq": -20.0,
        "source": str(data_dir / "train.bin"),
    }
    tmp_path = data_dir / "bigram_freq.pt.tmp"
    torch.save(obj, tmp_path)
    tmp_path.replace(out_path)
    volume_data.commit()
    return {
        "status": "created",
        "path": str(out_path),
        "n_bigrams": int(len(uniq)),
        "n_tokens": int(len(arr)),
    }


@app.local_entrypoint()
def main(dataset: str = "wikitext103"):
    info = prepare.remote(dataset)
    print(info)
