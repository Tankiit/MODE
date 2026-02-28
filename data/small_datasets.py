"""
Small datasets for debugging and quick experiments.

Options:
  - tinyshakespeare: ~1M tokens (~1MB)
  - wikitext2: ~2M tokens (~2MB)
  - openwebtext: sample up to 100M tokens

Usage:
    python data/small_datasets.py -d tinyshakespeare
    python data/small_datasets.py -d wikitext2
    python data/small_datasets.py -d openwebtext --samples 10000
"""

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset


def write_datafile(filename, toks):
    """
    Saves token data as a .bin file for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1  # version
    header[2] = len(toks)
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def get_tinyshakespeare():
    """Load TinyShakespeare dataset (~1M tokens)."""
    print("Loading TinyShakespeare...")
    dataset = load_dataset("karpathy/tinyshakespeare", split="train")
    text = "".join(dataset["text"])
    enc = tiktoken.get_encoding("gpt2")
    toks = enc.encode(text)
    print(f"Total tokens: {len(toks):,}")
    return toks


def get_wikitext2():
    """Load WikiText-2 dataset (~2M tokens)."""
    print("Loading WikiText-2...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    enc = tiktoken.get_encoding("gpt2")
    toks = enc.encode(text)
    print(f"Total tokens: {len(toks):,}")
    return toks


def get_openwebtext(samples=None):
    """Load OpenWebText dataset (optionally sample up to 100M tokens)."""
    print("Loading OpenWebText...")
    if samples is not None:
        dataset = load_dataset("Skylion007/openwebtext", split=f"train[:{samples}]")
    else:
        dataset = load_dataset("Skylion007/openwebtext", split="train")
    
    enc = tiktoken.get_encoding("gpt2")
    all_tokens = []
    for item in dataset:
        text = item["text"]
        toks = enc.encode(text)
        all_tokens.extend(toks)
    
    print(f"Total tokens: {len(all_tokens):,}")
    return all_tokens


def main():
    parser = argparse.ArgumentParser(description="Generate small dataset .bin files")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=["tinyshakespeare", "wikitext2", "openwebtext"],
                        help="Which dataset to use")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output filename (default: data/<dataset>.bin)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples to use (for openwebtext)")
    args = parser.parse_args()

    # Get tokens
    if args.dataset == "tinyshakespeare":
        toks = get_tinyshakespeare()
    elif args.dataset == "wikitext2":
        toks = get_wikitext2()
    elif args.dataset == "openwebtext":
        toks = get_openwebtext(args.samples)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, f"{args.dataset}.bin")

    # Write to file
    write_datafile(output_path, toks)
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    main()
