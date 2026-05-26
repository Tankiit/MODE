"""
One-time data preparation for SIEVE training.
Run this script once per dataset to create the memmap files and optional caches.

Usage:
    python prepare_sieve.py --dataset wikitext2
    python prepare_sieve.py --dataset wikitext2 --ref_model gpt2  # for S_E offline cache
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def prepare_dataset(
    dataset_name: str,
    data_dir: str,
    tokenizer_name: str = "gpt2",
    ref_model_name: Optional[str] = None,
    seq_len: int = 512,
):
    """
    Prepare dataset for SIEVE training.

    Creates:
    - train.bin / val.bin: uint16 token arrays
    - meta.json: vocab size, seq_len
    - freq.pt: log-IDF table for S_D scorer
    - ref_losses.bin: (optional) per-token reference losses for S_E scorer
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Preparing {dataset_name} dataset...")
    print(f"Output directory: {data_path}")

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    # Load and tokenize dataset
    if dataset_name == "wikitext2":
        from datasets import load_dataset
        print("Loading WikiText-2 from HuggingFace...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        # Tokenize
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=False,
                max_length=None,
                return_attention_mask=False,
            )

        print("Tokenizing...")
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

        # Build train/val splits
        train_tokens = []
        val_tokens = []

        for split in ["train", "validation"]:
            tokens = []
            for example in tokenized[split]:
                tokens.extend(example["input_ids"])

            if split == "train":
                train_tokens = tokens
            else:
                val_tokens = tokens

        print(f"Train tokens: {len(train_tokens)/1e6:.1f}M")
        print(f"Val tokens: {len(val_tokens)/1e6:.1f}M")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Save token arrays as memmap
    print("\nSaving token arrays...")
    for split_name, tokens in [("train", train_tokens), ("val", val_tokens)]:
        output_path = data_path / f"{split_name}.bin"
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(output_path)
        print(f"  {split_name}.bin: {len(arr)} tokens ({len(arr)*2/1e6:.1f} MB)")

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "dataset": dataset_name,
        "tokenizer": tokenizer_name,
    }
    with open(data_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved meta.json")

    # Compute frequency table for S_D scorer
    print("\nComputing token frequencies for S_D scorer...")
    all_tokens = np.array(train_tokens, dtype=np.int64)
    counts = np.bincount(all_tokens, minlength=vocab_size).astype(np.float32)
    # Log-IDF: log(N / (count + 1))
    log_idf = np.log((len(all_tokens) + 1) / (counts + 1))
    freq_tensor = torch.from_numpy(log_idf)
    torch.save(freq_tensor, data_path / "freq.pt")
    print(f"Saved freq.pt: shape={freq_tensor.shape}")

    # Optional: Precompute reference losses for S_E scorer
    if ref_model_name:
        print(f"\nComputing reference losses for S_E scorer using {ref_model_name}...")
        print("Loading reference model...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",  # Force CPU to avoid MPS issues
        )
        ref_model.eval()

        # Compute per-token losses for training set
        # This is memory-intensive, so we do it in chunks
        chunk_size = seq_len * 1000  # Process 1000 sequences at a time
        n_tokens = len(train_tokens)
        ref_losses = np.zeros(n_tokens, dtype=np.float32)

        print("Computing reference losses (this may take a while)...")
        with torch.no_grad():
            for start_idx in tqdm(range(0, n_tokens - seq_len, chunk_size), desc="Computing ref losses"):
                end_idx = min(start_idx + chunk_size + seq_len, n_tokens)
                chunk_tokens = train_tokens[start_idx:end_idx]

                # Create sequences
                n_seqs = (len(chunk_tokens) - 1) // seq_len
                if n_seqs == 0:
                    continue

                input_ids = torch.from_numpy(
                    np.array(chunk_tokens[:n_seqs * seq_len])
                ).reshape(n_seqs, seq_len)

                # Forward pass
                outputs = ref_model(input_ids=input_ids)
                logits = outputs.logits.float()

                # Compute per-token loss
                from torch.nn.functional import cross_entropy
                for i in range(n_seqs):
                    seq_logits = logits[i, :-1]  # [seq_len-1, vocab_size]
                    seq_labels = input_ids[i, 1:]  # [seq_len-1]
                    token_losses = cross_entropy(
                        seq_logits, seq_labels, reduction="none"
                    ).numpy()

                    # Store losses (first token has no loss)
                    loss_start = start_idx + i * seq_len
                    ref_losses[loss_start + 1:loss_start + seq_len] = token_losses

        # Save reference losses
        ref_losses.tofile(data_path / "ref_losses.bin")
        print(f"Saved ref_losses.bin: {len(ref_losses)} tokens ({len(ref_losses)*4/1e6:.1f} MB)")

    print(f"\n✓ Dataset preparation complete!")
    print(f"  Output directory: {data_path}")
    print(f"  Train tokens: {len(train_tokens)/1e6:.1f}M")
    print(f"  Val tokens: {len(val_tokens)/1e6:.1f}M")
    if ref_model_name:
        print(f"  Reference losses: precomputed")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for SIEVE training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "owm"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/wikitext2",
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name (must match model)",
    )
    parser.add_argument(
        "--ref_model",
        type=str,
        default=None,
        help="Reference model for precomputing S_E losses (e.g., gpt2)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Sequence length",
    )
    args = parser.parse_args()

    # Adjust data dir based on dataset
    if args.data_dir == "./data/wikitext2" and args.dataset != "wikitext2":
        args.data_dir = f"./data/{args.dataset}"

    prepare_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        tokenizer_name=args.tokenizer,
        ref_model_name=args.ref_model,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
