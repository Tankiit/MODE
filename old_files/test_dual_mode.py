#!/usr/bin/env python3
"""
Test Dual-MODE in isolation to debug issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import traceback

# Import from mode.py
import sys
sys.path.insert(0, '/Users/tanmoy/research/Dataset_Distillation/Coreset/LLM_stuff')

try:
    from mode import (Config, get_device, TokenModeMemoryStore,
                      ContextualizedMetaController, FeatureExtractor,
                      TextDataset, Trainer)
    print("✓ Successfully imported from mode.py")
except Exception as e:
    print(f"✗ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

def test_components():
    """Test individual components"""
    print("\n" + "="*70)
    print("TESTING DUAL-MODE COMPONENTS")
    print("="*70)

    config = Config()
    device = get_device()

    print(f"\n1. Testing TokenModeMemoryStore...")
    try:
        memory = TokenModeMemoryStore(config, device)
        print(f"   ✓ Memory store initialized (size: {memory.size()})")

        # Test adding
        test_context = torch.randn(config.n_embd, device=device)
        memory.add_context_token_pair(test_context, 42, 1.5)
        print(f"   ✓ Added test context (size: {memory.size()})")

        # Test retrieval
        score = memory.retrieve_mode_score(test_context, 42)
        print(f"   ✓ Retrieved score: {score:.4f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        traceback.print_exc()
        return False

    print(f"\n2. Testing ContextualizedMetaController...")
    try:
        controller = ContextualizedMetaController(config, device)
        print(f"   ✓ Meta-controller initialized")

        # Test prediction
        test_features = np.random.randn(config.n_embd)
        weights = controller.predict_weights(test_features)
        print(f"   ✓ Predicted weights: {weights}")
        print(f"   ✓ Sum: {weights.sum():.4f} (should be ~1.0)")

        # Test collecting training data
        test_scores = {
            'uncertainty': 0.5,
            'loss': 0.3,
            'coherence': 0.7,
            'diversity': 0.4,
            'token_mode': 0.2
        }
        controller.collect_training_pair(test_features, test_scores, 0.8)
        print(f"   ✓ Collected training pair")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        traceback.print_exc()
        return False

    print(f"\n3. Testing FeatureExtractor...")
    try:
        print(f"   Loading pretrained model...")
        model = AutoModelForCausalLM.from_pretrained('distilgpt2').to(device)
        ref_model = AutoModelForCausalLM.from_pretrained('distilgpt2').to('cpu')

        extractor = FeatureExtractor(model, ref_model, config, device)
        print(f"   ✓ Feature extractor initialized")

        # Test extraction
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        test_text = "This is a test sentence for feature extraction."
        inputs = tokenizer(test_text, return_tensors='pt', padding=True).to(device)

        features, context_bows = extractor.extract_all_features(
            inputs.input_ids,
            inputs.attention_mask,
            memory
        )

        print(f"   ✓ Extracted features:")
        for key, value in features.items():
            print(f"     - {key}: shape {value.shape}")
        print(f"   ✓ Context BOWs: {len(context_bows)} vectors")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        traceback.print_exc()
        return False

    print(f"\n✓ All components passed basic tests!")
    return True

def test_dual_mode_training():
    """Test Dual-MODE training with minimal data"""
    print("\n" + "="*70)
    print("TESTING DUAL-MODE TRAINING")
    print("="*70)

    # Create minimal config
    config = Config()
    config.train_samples = 20  # Minimal samples
    config.val_samples = 10
    config.epochs = 1  # Just 1 epoch
    config.batch_size = 2

    print(f"\nLoading minimal dataset...")
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', streaming=True)

        samples = []
        for sample in dataset:
            text = sample['text'].strip()
            if text and len(text) >= config.min_text_length:
                samples.append(sample)
            if len(samples) >= config.train_samples + config.val_samples:
                break

        train_samples = samples[:config.train_samples]
        val_samples = samples[config.train_samples:]

        print(f"✓ Loaded {len(train_samples)} train, {len(val_samples)} val samples")

        train_dataset = TextDataset(train_samples)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_texts = [s['text'] for s in val_samples]

    except Exception as e:
        print(f"✗ Data loading error: {e}")
        traceback.print_exc()
        return False

    print(f"\nInitializing Dual-MODE trainer...")
    try:
        trainer = Trainer(config, 'Test-Dual-MODE', 'mode')
        print(f"✓ Trainer initialized")

        print(f"\nRunning 1 training epoch...")
        result = trainer.train(train_loader, val_texts)

        print(f"\n✓ Training completed!")
        print(f"  Best Val PPL: {result['best_val_ppl']:.2f}")
        print(f"  Final Loss: {result['final_train_loss']:.4f}")

        return True

    except Exception as e:
        print(f"✗ Training error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DUAL-MODE DEBUG TEST")
    print("="*70)

    # Test components first
    if test_components():
        # If components work, test training
        test_dual_mode_training()
    else:
        print("\n✗ Component tests failed, skipping training test")

    print("\n" + "="*70)
    print("DEBUG TEST COMPLETE")
    print("="*70)
