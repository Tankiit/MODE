import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length',
                                   max_length=max_length, return_tensors='pt')

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

def compute_perplexity(model, dataloader, device):
    """Compute perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch['input_ids'])

            # Count non-padding tokens
            mask = batch['attention_mask']
            n_tokens = mask.sum().item()

            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    model.train()
    return perplexity

def taylor_expansion_score(model, batch, device):
    """
    Score batch using Taylor expansion approximation:
    L(θ + Δθ) ≈ L(θ) + ∇L(θ)·Δθ + (1/2)Δθ·H·Δθ

    We use gradient norm as a proxy for learning potential
    """
    model.train()
    model.zero_grad()

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Forward pass
    outputs = model(**batch, labels=batch['input_ids'])
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Compute gradient norm (first-order Taylor term)
    grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm()**2
    grad_norm = grad_norm.sqrt().item()

    # Higher gradient = more to learn from this batch
    score = grad_norm

    model.zero_grad()
    return score, loss.item()

def greats_training(model, candidate_batches, num_candidates=5, device='cpu'):
    """
    GREATS: Select batch with highest gradient norm (learning potential)
    """
    scores = []
    losses = []

    for batch in candidate_batches[:num_candidates]:
        score, loss = taylor_expansion_score(model, batch, device)
        scores.append(score)
        losses.append(loss)

    # Select batch with highest score
    best_idx = np.argmax(scores)
    best_batch = candidate_batches[best_idx]

    return best_batch, scores[best_idx], losses[best_idx]

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # Sample data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning has revolutionized computer vision.",
        "Natural language processing enables machines to understand text.",
        "Transformers are the foundation of modern language models.",
        "GPT stands for Generative Pre-trained Transformer.",
    ]

    # Create dataset
    dataset = SimpleTextDataset(texts, tokenizer)

    # Create test set for perplexity
    test_texts = [
        "Artificial intelligence is changing the world.",
        "Language models can generate human-like text.",
    ]
    test_dataset = SimpleTextDataset(test_texts, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=2)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_steps = 5
    num_candidates = 3

    print("\n=== Starting GREATS Training ===")
    print(f"Training steps: {num_steps}")
    print(f"Candidates per step: {num_candidates}\n")

    # Initial perplexity
    initial_ppl = compute_perplexity(model, test_loader, device)
    print(f"Initial perplexity: {initial_ppl:.2f}\n")

    for step in range(num_steps):
        # Generate candidate batches
        indices = np.random.choice(len(dataset), size=(num_candidates, 2), replace=True)
        candidate_batches = []
        for idx_list in indices:
            batch = {key: torch.stack([dataset[i][key] for i in idx_list])
                    for key in dataset[0].keys()}
            candidate_batches.append(batch)

        # Select best batch using GREATS
        best_batch, score, loss = greats_training(model, candidate_batches,
                                                  num_candidates, device)

        # Train on selected batch
        model.train()
        optimizer.zero_grad()
        best_batch = {k: v.to(device) for k, v in best_batch.items()}
        outputs = model(**best_batch, labels=best_batch['input_ids'])
        outputs.loss.backward()
        optimizer.step()

        # Compute perplexity
        ppl = compute_perplexity(model, test_loader, device)

        print(f"Step {step+1}/{num_steps} | "
              f"Score: {score:.2f} | "
              f"Loss: {loss:.4f} | "
              f"Perplexity: {ppl:.2f}")

    print(f"\nFinal perplexity: {ppl:.2f}")
    print(f"Perplexity change: {initial_ppl:.2f} → {ppl:.2f} "
          f"({((ppl-initial_ppl)/initial_ppl*100):+.1f}%)")

if __name__ == "__main__":
    main()
