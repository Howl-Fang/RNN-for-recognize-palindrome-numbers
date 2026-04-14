"""
Student Name: Placeholder Name
Student ID: 0000000
Student Email: student@example.com

Evaluation script for testing trained models on 12-digit palindromes
with uniform logarithmic distribution
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import json
import os

from model import PalindromeRNN, PalindromeDataset, get_device
from generate_data import generate_palindrome_dataset_log_uniform, is_palindrome
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_model(model_path: str, device: torch.device, max_length: int = 12) -> nn.Module:
    """Load a trained model."""
    model = PalindromeRNN(vocab_size=10, embedding_dim=16, hidden_dim=32, n_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_on_dataset(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, 
                        device: torch.device, max_length: int = 12) -> dict:
    """Evaluate model on given dataset."""
    
    # Create dataset
    test_dataset = PalindromeDataset(X_test, y_test, max_length=max_length)
    
    # Get predictions
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for x, y in test_dataset:
            x = x.unsqueeze(0).to(device)
            pred = model(x).squeeze().item()
            predictions.append(1 if pred > 0.5 else 0)
            true_labels.append(y.item())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'total_samples': int(len(predictions))
    }
    
    return results

def analyze_digit_distribution(X: np.ndarray) -> dict:
    """Analyze the distribution of digit counts in the dataset."""
    digit_counts = [len(str(x)) for x in X]
    unique, counts = np.unique(digit_counts, return_counts=True)
    
    distribution = {}
    for d, c in zip(unique, counts):
        distribution[str(d)] = int(c)
    
    return {
        'distribution': distribution,
        'mean_digits': float(np.mean(digit_counts)),
        'std_digits': float(np.std(digit_counts))
    }

def main():
    """Generate log-uniform test set and evaluate all models."""
    
    # Generate test set with log-uniform distribution
    print(f"\n{'='*70}")
    print("Generating test set with uniform logarithmic distribution (up to 12 digits)")
    print(f"{'='*70}\n")
    
    X_test, y_test = generate_palindrome_dataset_log_uniform(1000, max_digits=12, random_seed=99)
    
    # Save test set
    os.makedirs('test_sets', exist_ok=True)
    test_path = 'test_sets/test_12digit_logscale.csv'
    df = pd.DataFrame({'number': X_test, 'label': y_test})
    df.to_csv(test_path, index=False, header=False)
    print(f"Saved test set to {test_path}")
    
    # Analyze distribution
    dist_info = analyze_digit_distribution(X_test)
    print(f"\nDigit Count Distribution:")
    for digit, count in sorted(dist_info['distribution'].items()):
        print(f"  {digit} digits: {count} samples")
    print(f"  Mean: {dist_info['mean_digits']:.2f} digits")
    print(f"  Std: {dist_info['std_digits']:.2f} digits")
    
    # Count palindromes
    n_palindromes = np.sum(y_test)
    n_non_palindromes = len(y_test) - n_palindromes
    print(f"\nClass Distribution:")
    print(f"  Palindromes: {n_palindromes} ({100*n_palindromes/len(y_test):.1f}%)")
    print(f"  Non-palindromes: {n_non_palindromes} ({100*n_non_palindromes/len(y_test):.1f}%)")
    
    device = get_device()
    dataset_sizes = [200, 1000, 50000]
    all_results = {}
    
    print(f"\n{'='*70}")
    print("Evaluating models trained on 7-digit data on 12-digit log-scale data")
    print(f"{'='*70}\n")
    
    for size in dataset_sizes:
        model_path = f'models/model_{size}.pt'
        
        if not Path(model_path).exists():
            print(f"Model not found at {model_path}, skipping...")
            continue
        
        print(f"Model trained on {size} examples (7-digit):")
        model = load_model(model_path, device, max_length=12)
        results = evaluate_on_dataset(model, X_test, y_test, device, max_length=12)
        
        all_results[f'model_{size}_12digit'] = results
        
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TP={results['confusion_matrix']['true_positives']:4d}, "
              f"TN={results['confusion_matrix']['true_negatives']:4d}, "
              f"FP={results['confusion_matrix']['false_positives']:4d}, "
              f"FN={results['confusion_matrix']['false_negatives']:4d}")
        print()
    
    # Save evaluation results
    os.makedirs('results', exist_ok=True)
    result_path = 'results/evaluation_12digit_logscale.json'
    with open(result_path, 'w') as f:
        json.dump({
            'test_set_info': {
                'samples': len(X_test),
                'max_digits': 12,
                'distribution': 'uniform_logarithmic',
                'digit_distribution': dist_info
            },
            'model_results': all_results
        }, f, indent=2)
    print(f"Saved evaluation results to {result_path}")
    
    return all_results

if __name__ == '__main__':
    main()
